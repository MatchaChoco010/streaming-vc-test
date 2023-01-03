import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src.data.download as download

if __name__ == "__main__":
    download.download()
    download.extract()
