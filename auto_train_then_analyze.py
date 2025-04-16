import subprocess
import argparse
import datetime
import os

def run_batch_train(log_path=None):
    cmd = ["python", "batch_train.py"]
    print("🚀 Starting batch training...")

    with open(log_path, "w") if log_path else subprocess.DEVNULL as logfile:
        subprocess.run(cmd, stdout=logfile, stderr=subprocess.STDOUT, check=True)
    print("✅ Batch training completed.")

def run_batch_analysis(fast=True, skip_existing=True, skip_plots=False, workers=8, log_path=None):
    cmd = [
        "python", "analyze_all_checkpoints.py"
    ]
    print("\n🔬 Starting batch analysis...")
    with open(log_path, "w") if log_path else subprocess.DEVNULL as logfile:
        subprocess.run(cmd, stdout=logfile, stderr=subprocess.STDOUT, check=True)

    print("✅ Batch analysis completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_log", default=None, help="Optional path to save batch training logs")
    parser.add_argument("--analyze_log", default=None, help="Optional path to save analysis logs")
    parser.add_argument("--fast", action="store_true", help="Fast analysis (no gen or plot)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip already analyzed configs")
    parser.add_argument("--skip_plots", action="store_true", help="Skip plotting")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel threads for analysis")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_log = args.train_log or f"logs/batch_train_{timestamp}.log"
    analyze_log = args.analyze_log or f"logs/batch_analysis_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)

    run_batch_train(train_log)
    run_batch_analysis(
        fast=args.fast,
        skip_existing=args.skip_existing,
        skip_plots=args.skip_plots,
        workers=args.workers,
        log_path=analyze_log
    )

if __name__ == "__main__":
    main()
