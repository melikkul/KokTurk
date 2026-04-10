#!/bin/bash
# Download TTC-3600 raw text from a machine with internet access.
# Some restricted networks block raw.githubusercontent.com and Kaggle API requires auth.
#
# Run this on a machine with internet access:
#
# Option 1: Kaggle (requires API key at ~/.kaggle/kaggle.json)
#   pip install kaggle
#   kaggle datasets download -d savasy/ttc3600 -p ./ttc3600_raw/
#   unzip ./ttc3600_raw/ttc3600.zip -d ./ttc3600_raw/
#
# Option 2: Direct from GitHub
#   wget https://raw.githubusercontent.com/savasy/TurkishTextClassification/master/TTC3600.csv \
#     -O ./ttc3600_raw/TTC3600.csv
#
# Option 3: HuggingFace datasets
#   pip install datasets
#   python -c "
#   from datasets import load_dataset
#   ds = load_dataset('savasy/ttc3600')
#   ds['train'].to_csv('./ttc3600_raw/TTC3600.csv')
#   "
#
# Then copy to the project:
#   cp -r ./ttc3600_raw/ $PROJECT_DIR/data/external/ttc3600_raw/
#
# Expected format: CSV with columns "category" and "text"
# 3,600 documents, 6 categories (600 each):
#   ekonomi, kultursanat, saglik, siyaset, spor, teknoloji

echo "This script documents download methods for TTC-3600 raw text."
echo "Network restrictions may prevent automated download."
echo "Please run one of the methods above from a machine with full internet access."
