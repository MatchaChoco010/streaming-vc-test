import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src.data.prepare as prepare

if __name__ == "__main__":
    prepare.resample()
    prepare.copy_text()
