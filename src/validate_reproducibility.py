"""Lab 4 - Reproducibility check.

Trains the baseline twice with the same seed and verifies that all metrics are
identical (the original only checked that the script did not crash).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mlflow import train_and_track

print("Starting Reproducibility Validation...")
m1 = train_and_track(run_name="Reproducibility_Run_1")
m2 = train_and_track(run_name="Reproducibility_Run_2")

ok = True
for k in m1:
    same = abs(m1[k] - m2[k]) < 1e-12
    ok &= same
    print(f"{k:10s} run1={m1[k]:.6f} run2={m2[k]:.6f} -> {'MATCH' if same else 'DIFFERENT'}")

if not ok:
    print("Reproducibility validation FAILED: metrics differ between runs.")
    sys.exit(1)
print("Reproducibility validation completed successfully! Both runs gave identical metrics.")
