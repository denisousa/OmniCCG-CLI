# Note: this script runs OmniCCG locally without the CLI,
# but it requires a configuration file (config.json).

import json
from pathlib import Path
from core import execute_omniccg

def main():
    json_path = Path("../../config.json")

    with json_path.open("r", encoding="utf-8") as f:
        settings = json.load(f) 
    execute_omniccg(settings)

if __name__ == "__main__":
    main()
