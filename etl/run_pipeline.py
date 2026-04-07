"""
Full ETL pipeline orchestrator.
Runs all four stages in sequence: collect → bronze → silver → gold.

Usage:
    python -m etl.run_pipeline --start 2000-01-01 --end 2024-12-31
    python -m etl.run_pipeline --start 2025-01-01 --end 2025-12-31 --force
    python -m etl.run_pipeline --skip-collect  # skip ingestion, rebuild layers only
    python -m etl.run_pipeline --only pld      # collect only PLD, then rebuild layers
"""

import argparse
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run full Brazilian energy ETL pipeline")
    parser.add_argument("--start", default="2000-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--force", "-f", action="store_true", help="Re-download existing partitions")
    parser.add_argument("--skip-collect", action="store_true", help="Skip data collection step")
    parser.add_argument("--only", choices=["pld", "ons", "weather"],
                        help="Collect only one source")
    args = parser.parse_args()

    print("=" * 60)
    print("Brazilian Energy Price Forecast — Full ETL Pipeline")
    print("=" * 60)

    if not args.skip_collect:
        print("\nStep 1/4: Collecting data...")
        from etl.collect import CollectPLD, CollectONS, CollectWeather

        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        if not args.only or args.only == "pld":
            CollectPLD().process(start, end, force=args.force)
        if not args.only or args.only == "ons":
            CollectONS().process(start, end, force=args.force)
        if not args.only or args.only == "weather":
            CollectWeather().process(start, end, force=args.force)
    else:
        print("\nStep 1/4: Skipping data collection (--skip-collect).")

    print("\nStep 2/4: Building bronze layer...")
    from etl.bronze import build_bronze
    build_bronze()

    print("\nStep 3/4: Building silver layer (features)...")
    from etl.silver import build_silver
    build_silver()

    print("\nStep 4/4: Building gold layer (ABT + Feast prep)...")
    from etl.gold import build_gold
    build_gold()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    print("\nNext steps:")
    print("  cd feature_store && feast apply")
    print("  cd feature_store && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)")
    print("  python -m ml.train")
    print("  python -m ml.evaluate")
    print("  aim up --repo ./aim_logs --port 43800   # experiment tracking UI")
    print("  uvicorn serving.api:app --reload --port 8000")
    print("  streamlit run app/main.py")


if __name__ == "__main__":
    main()
