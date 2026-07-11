"""
gex_publisher.py

Run this once a day (cron / Task Scheduler / manual) OUTSIDE of QuantConnect.
It computes GEX levels via gex_engine.py and appends one row to a historical
CSV per product. Those CSVs are what the QC custom data class (gex_data.py)
reads -- QC backtests need full history, so these files must accumulate
over time, not just hold today's row.

Host the resulting CSVs somewhere QC's cloud can pull from over plain HTTPS
(a public GitHub raw URL is the easiest free option -- commit-and-push
after each run). Dropbox (dl=1 link) and S3 public buckets also work.

Two products, run independently since ES and NQ need separate underlyings:
    ES (S&P futures)    <- SPY options chain  -> gex_levels_spy.csv
    NQ (Nasdaq futures) <- QQQ options chain  -> gex_levels_qqq.csv
Using SPY-derived levels for NQ (or vice versa) would be wrong -- they're
different indices with different gamma structure.

Row schema (headered CSV, one row per trading day):
date,spot,total_net_gex,gamma_flip,call_res_1,call_res_2,call_res_3,
put_sup_1,put_sup_2,put_sup_3,largest_wall,regime
"""

from __future__ import annotations

import os
import csv
import datetime as dt
from typing import Optional

from gex_engine import GEXEngine, GEXConfig, ChainFetcher, GEXLevels

CSV_HEADER = [
    "date", "spot", "total_net_gex", "gamma_flip",
    "call_res_1", "call_res_2", "call_res_3",
    "put_sup_1", "put_sup_2", "put_sup_3",
    "largest_wall", "regime",
]

# Registry of everything run_all() drives. Add a product here and both
# run_daily_gex.py and gex_data.py's SYMBOL_TO_URL map pick it up --
# see the "adding a third product" note in run_daily_gex.py.
PRODUCTS = {
    "ES": {"ticker": "SPY", "csv": "gex_levels_spy.csv"},
    "NQ": {"ticker": "QQQ", "csv": "gex_levels_qqq.csv"},
}


def _pad3(vals: list, fill=None) -> list:
    vals = list(vals)[:3]
    while len(vals) < 3:
        vals.append(fill)
    return vals


def levels_to_row(levels: GEXLevels, as_of: Optional[dt.date] = None) -> list:
    as_of = as_of or dt.date.today()
    call_res = _pad3(levels.call_resistance)
    put_sup = _pad3(levels.put_support)
    return [
        as_of.isoformat(),
        levels.spot,
        levels.total_net_gex,
        levels.gamma_flip if levels.gamma_flip is not None else "",
        *[c if c is not None else "" for c in call_res],
        *[p if p is not None else "" for p in put_sup],
        levels.largest_wall,
        levels.regime,
    ]


def append_daily_row(csv_path: str, levels: GEXLevels, as_of: Optional[dt.date] = None) -> None:
    """Idempotent-ish append: replaces today's row if the script is re-run
    same day, otherwise appends. Keeps the file sorted by date."""
    as_of = as_of or dt.date.today()
    row = levels_to_row(levels, as_of)

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if r and r[0] != row[0]:  # drop any existing row for today, keep rest
                    rows.append(r)

    rows.append([str(x) for x in row])
    rows.sort(key=lambda r: r[0])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def run_daily(ticker: str = "SPY", csv_path: str = "gex_levels_spy.csv",
              max_dte: int = 45) -> GEXLevels:
    fetcher = ChainFetcher(ticker)
    raw = fetcher.fetch(max_dte=max_dte)
    if raw.empty:
        raise ValueError(
            f"Fetched an empty options chain for '{ticker}' (0 contracts within "
            f"max_dte={max_dte}). Not writing a row -- check ticker/market hours/network."
        )

    engine = GEXEngine(GEXConfig())
    priced = engine.price_chain(raw)
    if priced.empty:
        raise ValueError(
            f"All {len(raw)} contracts were dropped during pricing (IV solve failed "
            f"or filtered by min_oi) -- no valid data to compute GEX from. This usually "
            f"means stale/crossed quotes across the whole chain; check the raw feed."
        )

    spot = float(raw["spot"].iloc[0])
    levels = engine.key_levels(priced, spot=spot)
    append_daily_row(csv_path, levels)
    return levels


def run_all(products: dict = PRODUCTS, max_dte: int = 45) -> dict:
    """
    Runs run_daily() for every product in the registry. One product failing
    (e.g. QQQ chain temporarily unavailable) does NOT stop the others --
    each is isolated so you still get a good ES update even if the NQ leg
    breaks that day.

    Returns {product_name: GEXLevels | Exception} so the caller can decide
    what to do with partial failures.
    """
    results = {}
    for name, cfg in products.items():
        print(f"--- {name} ({cfg['ticker']}) ---")
        try:
            levels = run_daily(ticker=cfg["ticker"], csv_path=cfg["csv"], max_dte=max_dte)
            print(f"  OK: regime={levels.regime} flip={levels.gamma_flip} "
                  f"total_net_gex={levels.total_net_gex:,.0f}")
            results[name] = levels
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = e
    return results


# --------------------------------------------------------------------------
# Self-test: validates the CSV round trip (write -> parse) without touching
# the network, using the same synthetic chain from gex_engine.py.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from gex_engine import _synthetic_chain

    print("Building levels from synthetic chain and writing a test CSV...\n")
    raw = _synthetic_chain()
    engine = GEXEngine(GEXConfig())
    priced = engine.price_chain(raw)
    levels = engine.key_levels(priced, spot=580.0)

    test_path = "/tmp/gex_levels_test.csv"
    if os.path.exists(test_path):
        os.remove(test_path)

    append_daily_row(test_path, levels, as_of=dt.date(2026, 6, 30))
    append_daily_row(test_path, levels, as_of=dt.date(2026, 7, 1))
    # re-run "today" to confirm idempotent replace, not duplicate append
    append_daily_row(test_path, levels, as_of=dt.date(2026, 7, 1))

    with open(test_path) as f:
        content = f.read()
    print(content)

    n_data_rows = content.strip().count("\n")  # header + 2 unique dates expected
    assert n_data_rows == 2, f"expected 2 data rows after idempotent re-run, got {n_data_rows}"
    print("OK: idempotent daily append behaves correctly (2 unique dates, no duplicate).")
