"""Lab 6 pipeline. Extra arguments are passed to src/train_registry.py, e.g.
    python pipelines/run_lab6_registry.py
    python pipelines/run_lab6_registry.py --run-name RandomForest_Registry_V2 --n-estimators 200 --max-depth 5
"""
import subprocess
import sys


def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 6: Model Registry & Lifecycle")
    print("[INFO] =========================================")
    scripts = [
        ["src/train_registry.py"] + sys.argv[1:],   # trains and registers a new model version
        ["src/automate_lifecycle.py"],              # compares versions and promotes the Champion
        ["src/generate_registry_report.py"],        # writes the JSON deployment artifact
    ]
    for cmd in scripts:
        print(f"\n[INFO] ---> Executing {' '.join(cmd)}...")
        if subprocess.run([sys.executable] + cmd).returncode != 0:
            print(f"[ERROR] Pipeline halted. {cmd[0]} caught an error.")
            sys.exit(1)
    print("\n[SUCCESS] Lab 6 Model Registry Pipeline fully executed!")


if __name__ == "__main__":
    execute_pipeline()
