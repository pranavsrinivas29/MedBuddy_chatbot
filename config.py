# backend/config.py

import os
from pathlib import Path

# Project root directory

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_FILE = DATA_DIR / "drugs_side_effects_drugs_com.csv"
CHROMA_DB_DIR = DATA_DIR / "chroma_store"
print("Resolved CSV path:", CSV_FILE)
print("File exists:", CSV_FILE.exists())
