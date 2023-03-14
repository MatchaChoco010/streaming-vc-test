import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.fit_km import fit

if __name__ == "__main__":
    fit()
