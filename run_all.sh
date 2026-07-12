#!/usr/bin/env bash
set -e
python scripts/01_cell_typing.py
python scripts/02_feature_engineering.py
python scripts/03_modeling.py
python scripts/04_make_figures.py
echo "Pipeline complete. See results/tables/ and results/figures/"
