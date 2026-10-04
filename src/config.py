"""Run settings, read once on import.

Set the environment variables before importing src, or edit the defaults below.
The notebooks set these values in their cells; see CODE_PROVENANCE.md for the
source mappings.
"""
import os

SEED       = int(os.environ.get("TRM_SEED", 0))          # Recorded seeds: 0, 40, 100
NUM_DIGITS = int(os.environ.get("TRM_NUM_DIGITS", 4))    # Recorded runs use 4 or 8 digits
EPOCHS     = int(os.environ.get("TRM_EPOCHS", 100))      # 200 for the extended 8-digit runs
TARGET     = os.environ.get("TRM_TARGET", "softmean")    # "softmean", "geomean", or "binary_em"
