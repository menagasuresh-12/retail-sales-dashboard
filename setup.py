"""
setup.py  —  One-shot setup script
-----------------------------------
Run this ONCE to:
  1. Generate raw dataset
  2. Clean it
  3. Train the ML model
  4. Save everything to /data/

Then launch the dashboard with:
    streamlit run app/dashboard.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

steps = [
    ("Generating dataset...",  [sys.executable, str(ROOT / "src/generate_data.py")]),
    ("Cleaning data...",       [sys.executable, str(ROOT / "src/data_processing.py")]),
    ("Training ML model...",   [sys.executable, str(ROOT / "src/ml_model.py")]),
]

for msg, cmd in steps:
    print(f"\n{'='*50}")
    print(f"  {msg}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"ERROR: step failed. Fix the error above and re-run.")
        sys.exit(1)

print("\n" + "="*50)
print("  ✅ Setup complete!")
print("="*50)
print("\nLaunch the dashboard with:")
print("  streamlit run app/dashboard.py\n")
