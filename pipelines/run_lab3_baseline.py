import subprocess
import sys


def execute_pipeline():
    print("[INFO] Starting Lab 3: Baseline ML Pipeline")
    for script in ["src/preprocess.py", "src/train.py", "src/evaluate.py"]:
        print(f"\n[INFO] ---> Executing {script}...")
        if subprocess.run([sys.executable, script]).returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed.")
            sys.exit(1)
    print("\n[SUCCESS] Lab 3 Pipeline fully executed!")


if __name__ == "__main__":
    execute_pipeline()
