"""
gex_engine.py

Standalone GEX (Gamma Exposure) calculation engine for SPX/SPY-style
index/ETF options, intended as a pre-processing step that feeds levels
into a QuantConnect LEAN futures algo (ES/NQ).

Pricing model: Black-Scholes-Merton (continuous dividend yield), which is
correct for SPY/SPX listed options. If you later want native ES/NQ
futures-options gamma, swap the pricing block for Black-76 -- see the
`black76_gamma` stub at the bottom for the formula shape.

Pipeline:
    1. ChainFetcher     -> pulls raw chain (yfinance) into a flat DataFrame
    2. GEXEngine.price_chain(...)   -> fills/repairs IV, computes per-contract gamma & $GEX
    3. GEXEngine.aggregate_by_strike(...) -> net dealer GEX per strike
    4. GEXEngine.find_gamma_flip(...)     -> zero-gamma / HVL price level
    5. GEXEngine.key_levels(...)          -> call resistance / put support / flip, packaged

No live network calls happen at import time. Everything is testable with
synthetic data (see `__main__` block) even with no internet access.
"""

from __future__ import annotations

import math
import json
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


# --------------------------------------------------------------------------
# Black-Scholes-Merton primitives (equity/ETF options, continuous dividend q)
# --------------------------------------------------------------------------

def bsm_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
              option_type: Literal["C", "P"]) -> float:
    """Black-Scholes-Merton price. T in years, r/q/sigma annualized decimals."""
    if T <= 0 or sigma <= 0:
        # intrinsic value fallback for expired/degenerate inputs
        return max(0.0, (S - K) if option_type == "C" else (K - S))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "C":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bsm_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Gamma is identical for calls and puts under BSM."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float,
                 option_type: Literal["C", "P"],
                 lo: float = 1e-4, hi: float = 5.0) -> Optional[float]:
    """
    Back out IV from a market price via Brent's method. Returns None if it
    fails to bracket a root (e.g. price outside no-arbitrage bounds -- common
    for deep ITM/far OTM or stale/crossed quotes; caller should drop these).
    """
    if price <= 0 or T <= 0:
        return None

    def f(sigma):
        return bsm_price(S, K, T, r, q, sigma, option_type) - price

    try:
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            return None
        return brentq(f, lo, hi, maxiter=200, xtol=1e-6)
    except (ValueError, RuntimeError):
        return None


# --------------------------------------------------------------------------
# Chain fetching (yfinance). Isolated so the data source is swappable later
# (Polygon/EODHD/Tradier/CME etc.) without touching the pricing/aggregation.
# --------------------------------------------------------------------------

class ChainFetcher:
    """Pulls a full options chain (all strikes x expirations) via yfinance."""

    def __init__(self, ticker: str):
        self.ticker = ticker

    def fetch(self, max_dte: Optional[int] = None) -> pd.DataFrame:
        """
        Returns a flat DataFrame with columns:
        expiration, dte, strike, option_type, open_interest, bid, ask, mid,
        last, yf_implied_vol, spot
        `max_dte` optionally caps how far out to pull (e.g. 60 to skip LEAPS).
        """
        import yfinance as yf  # imported lazily so the module loads without it installed

        tk = yf.Ticker(self.ticker)
        hist = tk.history(period="1d")
        if hist.empty:
            raise ValueError(
                f"yfinance returned no price history for '{self.ticker}' -- "
                f"check the ticker symbol and your network connection."
            )
        spot = hist["Close"].iloc[-1]
        today = dt.date.today()

        expirations = tk.options
        if not expirations:
            raise ValueError(
                f"yfinance returned no option expirations for '{self.ticker}' -- "
                f"this ticker may not have listed options."
            )

        rows = []
        for exp_str in expirations:
            exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte < 0:
                continue
            if max_dte is not None and dte > max_dte:
                continue

            try:
                chain = tk.option_chain(exp_str)
            except Exception as e:
                # one bad/throttled expiration shouldn't kill the whole fetch
                print(f"  [ChainFetcher] skipping expiration {exp_str}: {e}")
                continue

            for opt_type, df in (("C", chain.calls), ("P", chain.puts)):
                for _, row in df.iterrows():
                    bid = row.get("bid", np.nan)
                    ask = row.get("ask", np.nan)
                    bid = np.nan if bid is None else bid
                    ask = np.nan if ask is None else ask
                    last = row.get("lastPrice", np.nan)
                    last = np.nan if last is None else last

                    mid = np.nanmean([bid, ask]) if not (pd.isna(bid) and pd.isna(ask)) else last
                    rows.append({
                        "expiration": exp_str,
                        "dte": dte,
                        "strike": row["strike"],
                        "option_type": opt_type,
                        "open_interest": row.get("openInterest", 0) or 0,
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "last": last,
                        "yf_implied_vol": row.get("impliedVolatility", np.nan),
                        "spot": spot,
                    })

        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# GEX engine
# --------------------------------------------------------------------------

@dataclass
class GEXConfig:
    contract_multiplier: int = 100
    risk_free_rate: float = 0.045     # update from current SOFR/T-bill
    dividend_yield: float = 0.013     # update from current SPY trailing yield
    dealer_short_puts: bool = True    # True: dealer GEX = CallGEX - PutGEX (standard convention)
    trust_source_iv: bool = False     # if True, use yf_implied_vol when sane; else always re-solve
    min_oi: int = 1                   # drop zero-OI contracts before aggregating
    trading_days_per_year: float = 252.0
    dte_convention: Literal["calendar", "trading"] = "calendar"


class GEXEngine:

    def __init__(self, config: Optional[GEXConfig] = None):
        self.cfg = config if config is not None else GEXConfig()

    # ---- step 1: price / repair IV, compute gamma & per-contract $GEX ----

    def price_chain(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cfg = self.cfg

        days_per_year = 365.0 if cfg.dte_convention == "calendar" else cfg.trading_days_per_year
        df["T"] = df["dte"].clip(lower=0) / days_per_year

        ivs = []
        for _, row in df.iterrows():
            iv = None
            if cfg.trust_source_iv:
                src_iv = row.get("yf_implied_vol", np.nan)
                if pd.notna(src_iv) and 0.01 < src_iv < 5.0:
                    iv = src_iv
            if iv is None:
                price = row["mid"] if pd.notna(row["mid"]) and row["mid"] > 0 else row["last"]
                if pd.notna(price) and price > 0:
                    iv = implied_vol(
                        price=price, S=row["spot"], K=row["strike"], T=row["T"],
                        r=cfg.risk_free_rate, q=cfg.dividend_yield,
                        option_type=row["option_type"],
                    )
            ivs.append(iv)
        df["iv"] = ivs

        # drop contracts we couldn't price and low/no-OI noise
        df = df[df["iv"].notna() & (df["open_interest"] >= cfg.min_oi)].copy()

        # vectorized BSM gamma (avoids a slow row-wise .apply over the full chain)
        S, K, T, sigma = df["spot"].values, df["strike"].values, df["T"].values, df["iv"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(S / K) + (cfg.risk_free_rate - cfg.dividend_yield + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            gamma = np.exp(-cfg.dividend_yield * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        df["gamma"] = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

        # standard dollar-gamma-per-1%-move convention
        df["contract_gex"] = (
            df["gamma"] * df["open_interest"] * cfg.contract_multiplier * (df["spot"] ** 2) * 0.01
        )
        # sign convention: puts flipped negative under the "dealer short puts, long calls" assumption
        sign = df["option_type"].map({"C": 1, "P": -1 if cfg.dealer_short_puts else 1})
        df["signed_gex"] = df["contract_gex"] * sign

        return df

    # ---- step 2: aggregate to strike level ----

    def aggregate_by_strike(self, priced_df: pd.DataFrame) -> pd.DataFrame:
        # split call/put contract_gex into separate columns first so the
        # groupby-sum is a plain, robust aggregation (no index-alignment
        # tricks that break if priced_df's index has duplicates/gaps)
        tmp = priced_df.copy()
        tmp["call_gex_"] = np.where(tmp["option_type"] == "C", tmp["contract_gex"], 0.0)
        tmp["put_gex_"] = np.where(tmp["option_type"] == "P", tmp["contract_gex"], 0.0)

        agg = (
            tmp.groupby("strike", as_index=False)
            .agg(call_gex=("call_gex_", "sum"),
                 put_gex=("put_gex_", "sum"),
                 net_gex=("signed_gex", "sum"),
                 total_oi=("open_interest", "sum"))
            .sort_values("strike")
            .reset_index(drop=True)
        )
        return agg

    def total_net_gex(self, priced_df: pd.DataFrame) -> float:
        return float(priced_df["signed_gex"].sum())

    # ---- step 3: gamma flip / HVL -----------------------------------------
    # Zero-gamma level: recompute total net GEX at a grid of hypothetical
    # spot prices (holding IV/OI fixed -- the standard simplifying
    # assumption used by SpotGamma-style models) and find where it crosses 0.

    def find_gamma_flip(self, priced_df: pd.DataFrame, spot: float,
                         pct_range: float = 0.10, steps: int = 200,
                         _widened: bool = False) -> Optional[float]:
        cfg = self.cfg
        K = priced_df["strike"].values
        T = priced_df["T"].values
        sigma = priced_df["iv"].values
        oi = priced_df["open_interest"].values
        r, q = cfg.risk_free_rate, cfg.dividend_yield
        # invariant across the grid loop -- compute once, not per iteration
        sign = np.where(priced_df["option_type"].values == "C", 1, -1 if cfg.dealer_short_puts else 1)

        grid = np.linspace(spot * (1 - pct_range), spot * (1 + pct_range), steps)
        totals = []
        for s in grid:
            # vectorized BSM gamma across all contracts at hypothetical spot s
            with np.errstate(divide="ignore", invalid="ignore"):
                d1 = (np.log(s / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                gamma = np.exp(-q * T) * norm.pdf(d1) / (s * sigma * np.sqrt(T))
            gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
            contract_gex = gamma * oi * cfg.contract_multiplier * (s ** 2) * 0.01
            totals.append(np.sum(contract_gex * sign))
        totals = np.array(totals)

        sign_changes = np.where(np.diff(np.sign(totals)) != 0)[0]
        if len(sign_changes) == 0:
            if not _widened:
                # try once more with a wider net before giving up -- a flip
                # right at the edge of +/-10% is common in strongly
                # one-sided regimes
                return self.find_gamma_flip(priced_df, spot, pct_range=pct_range * 3,
                                             steps=steps, _widened=True)
            return None  # genuinely no flip nearby -- market is deep in one regime
        i = sign_changes[0]
        # linear interpolation between grid[i] and grid[i+1]
        x0, x1 = grid[i], grid[i + 1]
        y0, y1 = totals[i], totals[i + 1]
        return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))

    # ---- step 4: key levels bundle ----------------------------------------

    def key_levels(self, priced_df: pd.DataFrame, spot: float, top_n: int = 3) -> "GEXLevels":
        by_strike = self.aggregate_by_strike(priced_df)
        flip = self.find_gamma_flip(priced_df, spot)

        # call_gex / put_gex are unsigned magnitudes (gamma*OI*mult*spot^2 is
        # always >= 0), so "biggest wall" on either side means nlargest, not
        # nsmallest. (nsmallest would surface near-zero, low-OI noise strikes
        # instead of the actual concentration walls.)
        call_res = by_strike.nlargest(top_n, "call_gex")["strike"].tolist()
        put_sup = by_strike.nlargest(top_n, "put_gex")["strike"].tolist()
        biggest_wall = by_strike.loc[by_strike["net_gex"].abs().idxmax(), "strike"]
        total = self.total_net_gex(priced_df)

        return GEXLevels(
            spot=spot,
            total_net_gex=total,
            gamma_flip=flip,
            call_resistance=call_res,
            put_support=put_sup,
            largest_wall=float(biggest_wall),
            regime="positive" if total > 0 else "negative",
            by_strike=by_strike,
        )


@dataclass
class GEXLevels:
    spot: float
    total_net_gex: float
    gamma_flip: Optional[float]
    call_resistance: list
    put_support: list
    largest_wall: float
    regime: str
    by_strike: pd.DataFrame

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize the scalar levels (not the full by-strike table) for
        consumption by a QuantConnect algo via ObjectStore or a data file."""
        payload = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "spot": self.spot,
            "total_net_gex": self.total_net_gex,
            "gamma_flip": self.gamma_flip,
            "call_resistance": self.call_resistance,
            "put_support": self.put_support,
            "largest_wall": self.largest_wall,
            "regime": self.regime,
        }
        text = json.dumps(payload, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text


# --------------------------------------------------------------------------
# Black-76 stub (only needed if you switch to native CME options-on-futures)
# --------------------------------------------------------------------------

def black76_gamma(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Gamma for options on a futures contract (Black-76). F = futures price."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return math.exp(-r * T) * norm.pdf(d1) / (F * sigma * math.sqrt(T))


# --------------------------------------------------------------------------
# Self-test with synthetic data -- runs with zero network access.
# --------------------------------------------------------------------------

def _synthetic_chain(spot=580.0, n_strikes=21, dte=30) -> pd.DataFrame:
    """Builds a fake chain by pricing known-IV options with BSM, then only
    keeping price/OI (as if that's all a real feed gave us) so price_chain()
    has to re-solve IV -- this validates the whole round trip."""
    r, q = 0.045, 0.013
    T = dte / 365.0
    strikes = np.linspace(spot * 0.85, spot * 1.15, n_strikes)
    rows = []
    rng = np.random.default_rng(7)
    for k in strikes:
        true_iv = 0.14 + 0.05 * abs(k - spot) / spot  # rough smile
        for opt_type in ("C", "P"):
            price = bsm_price(spot, k, T, r, q, true_iv, opt_type)
            oi = int(rng.integers(100, 20000) * math.exp(-((k - spot) ** 2) / (2 * (spot * 0.05) ** 2)))
            rows.append({
                "expiration": (dt.date.today() + dt.timedelta(days=dte)).isoformat(),
                "dte": dte, "strike": k, "option_type": opt_type,
                "open_interest": oi, "bid": price * 0.98, "ask": price * 1.02,
                "mid": price, "last": price, "yf_implied_vol": np.nan, "spot": spot,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running self-test on synthetic chain (no network required)...\n")

    raw = _synthetic_chain()
    engine = GEXEngine(GEXConfig(dividend_yield=0.013, risk_free_rate=0.045))
    priced = engine.price_chain(raw)

    print(f"Priced {len(priced)}/{len(raw)} contracts (dropped = failed IV solve or OI filter)")
    print(f"IV solve sanity: min={priced['iv'].min():.3f} max={priced['iv'].max():.3f}")

    levels = engine.key_levels(priced, spot=580.0)
    print("\n--- GEX levels ---")
    print(f"Spot:              {levels.spot}")
    print(f"Total net GEX:     {levels.total_net_gex:,.0f}")
    print(f"Regime:            {levels.regime}")
    print(f"Gamma flip (HVL):  {levels.gamma_flip}")
    print(f"Call resistance:   {levels.call_resistance}")
    print(f"Put support:       {levels.put_support}")
    print(f"Largest |wall|:    {levels.largest_wall}")

    print("\n--- by-strike (head) ---")
    print(levels.by_strike.head(8).to_string(index=False))

    print("\n--- JSON export preview ---")
    print(levels.to_json())
