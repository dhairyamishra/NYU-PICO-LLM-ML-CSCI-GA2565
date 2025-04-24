import subprocess
import sys

def run_script(path, args=None):
    cmd = [sys.executable, path]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Script failed: {path}")
        sys.exit(result.returncode)

def main():
    print("🚀 Step 1: Running batch_train.py")
    run_script("batch_train.py")

    print("\n📊 Step 2: Running full_analysis_pipeline.py")
    run_script("full_analysis_pipeline.py", ["--run_checkpoints"])

    print("\n✅ All steps complete!")

if __name__ == "__main__":
    main()
