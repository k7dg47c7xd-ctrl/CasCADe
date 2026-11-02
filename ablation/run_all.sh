#!/usr/bin/env bash
set -euo pipefail

# Repair-stage paper ablations only. Difficulties are read from config.yaml
# (currently simple + complex). This script does not retrain models.

python run_ablation.py --group repair   --difficulty all
python run_ablation.py --group trigger  --difficulty all
python run_ablation.py --group feedback --difficulty all
python summarize_all.py

echo "DONE"
