"""
run_daily_gex.py

The ONE thing you run each day. It:
  1. Computes today's GEX levels for BOTH products (ES<-SPY, NQ<-QQQ)
  2. Appends/updates today's row in each product's CSV
  3. Commits and pushes whichever CSVs actually got new data

If one product fails (e.g. QQQ chain temporarily unavailable), the other
still gets computed, written, and pushed -- see gex_publisher.run_all()
for the isolation logic. You'll get a clear FAILED line for the broken one
without losing the working one.

--------------------------------------------------------------------------
ONE-TIME SETUP (do this once, not daily):
--------------------------------------------------------------------------
1. Create a GitHub repo (can be public, e.g. "gex-data").
2. Clone it locally, e.g.:
       git clone https://github.com/<you>/gex-data.git
3. Copy gex_engine.py, gex_publisher.py, and this file into that cloned
   folder.
4. Edit REPO_DIR below to point at that folder's absolute path.
5. Make sure `git` is installed and you've pushed to this repo manually at
   least once before (so your credentials/SSH key are already set up).
6. In your QC gex_data.py, set SYMBOL_TO_URL to point at both raw URLs:
       https://raw.githubusercontent.com/<you>/gex-data/main/gex_levels_spy.csv
       https://raw.githubusercontent.com/<you>/gex-data/main/gex_levels_qqq.csv

--------------------------------------------------------------------------
DAILY USAGE:
--------------------------------------------------------------------------
    python run_daily_gex.py

--------------------------------------------------------------------------
AUTOMATING IT (so you never have to remember):
--------------------------------------------------------------------------
Mac/Linux (cron) -- run `crontab -e` and add a line like:
    0 17 * * 1-5 cd /path/to/gex-data && /usr/bin/python3 run_daily_gex.py >> run_daily_gex.log 2>&1

Windows (Task Scheduler):
    Create a Basic Task -> Trigger: Daily, weekdays, after market close ->
    Action: "Start a program" -> Program: python.exe ->
    Arguments: run_daily_gex.py -> Start in: (this folder's path)

--------------------------------------------------------------------------
ADDING A THIRD PRODUCT LATER (e.g. RTY <- IWM):
--------------------------------------------------------------------------
Add one entry to PRODUCTS in gex_publisher.py, e.g.
    "RTY": {"ticker": "IWM", "csv": "gex_levels_iwm.csv"}
Everything else (this script, the git push, the QC symbol routing in
gex_data.py) picks it up automatically as long as you also add the matching
entry to SYMBOL_TO_URL in gex_data.py and subscribe to it in Initialize().
"""

from __future__ import annotations

import subprocess
import sys
import traceback
import datetime as dt
from pathlib import Path

# ---- EDIT THIS: absolute path to your cloned GitHub repo folder ----
REPO_DIR = Path(__file__).resolve().parent  # defaults to "wherever this script lives"


def run(cmd: list, cwd: Path, allow_fail: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(f"$ {' '.join(cmd)}")
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if result.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def main():
    print(f"=== GEX daily run: {dt.datetime.now().isoformat(timespec='seconds')} ===\n")

    if dt.date.today().weekday() >= 5 and "--force" not in sys.argv:
        print("Today is a weekend -- markets are closed, skipping to avoid writing a "
              "mislabeled/duplicate row. Run with --force to override.")
        sys.exit(0)

    # 1. compute today's levels for every product and update each CSV
    sys.path.insert(0, str(REPO_DIR))
    from gex_publisher import run_all, PRODUCTS, GEXLevels

    # PRODUCTS gives relative csv filenames; resolve them against REPO_DIR
    # so this script works regardless of what directory it's launched from
    # (important for cron, which doesn't run with your normal shell cwd).
    products = {
        name: {"ticker": cfg["ticker"], "csv": str(REPO_DIR / cfg["csv"])}
        for name, cfg in PRODUCTS.items()
    }

    results = run_all(products)
    print()

    succeeded = [name for name, r in results.items() if isinstance(r, GEXLevels)]
    failed = [name for name, r in results.items() if not isinstance(r, GEXLevels)]

    if failed:
        print(f"FAILED products: {failed} -- see FAILED lines above for details.")
        print("--- traceback for the last failure (for debugging) ---")
        last_failed = results[failed[-1]]
        traceback.print_exception(type(last_failed), last_failed, last_failed.__traceback__)
        print("---")

    if not succeeded:
        print("\nNo products succeeded -- nothing new to push. Check data source / network.")
        sys.exit(1)

    # 2. commit and push whatever CSVs actually changed
    is_repo = (REPO_DIR / ".git").exists()
    if not is_repo:
        print(f"\n{REPO_DIR} is not a git repo -- skipping commit/push. "
              f"See the setup instructions at the top of this file.")
        sys.exit(0 if not failed else 1)

    csv_filenames = [PRODUCTS[name]["csv"] for name in succeeded]
    run(["git", "add", *csv_filenames], cwd=REPO_DIR)

    commit_msg = f"GEX levels {dt.date.today().isoformat()} ({', '.join(succeeded)})"
    commit_result = run(["git", "commit", "-m", commit_msg], cwd=REPO_DIR, allow_fail=True)
    if "nothing to commit" in (commit_result.stdout + commit_result.stderr).lower():
        print("\nNo changes to commit (already up to date for today).")
        sys.exit(0 if not failed else 1)

    push_result = run(["git", "push"], cwd=REPO_DIR, allow_fail=True)
    if push_result.returncode != 0:
        print("\nFAILED to push. The commit was made locally, but didn't reach GitHub.")
        print("Common causes: no remote configured, needs re-auth (token expired), or no internet.")
        print("Fix manually with: git push   (from inside your repo folder)")
        sys.exit(1)

    print(f"\nDone -- pushed today's GEX levels for: {succeeded}")
    if failed:
        print(f"(but {failed} failed to compute this run -- re-run later or check logs above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
