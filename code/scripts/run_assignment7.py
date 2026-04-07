from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_opt.analysis.assignment7 import run_assignment7_rolling_horizon
from battery_opt.config import load_case_config


def main() -> None:
    case_config = load_case_config(PROJECT_ROOT / "configs" / "base_case.json")
    output_dir = PROJECT_ROOT / "outputs" / "assignment7"
    run_assignment7_rolling_horizon(case_config, output_dir)
    print(f"Wrote assignment 7 outputs to {output_dir}")


if __name__ == "__main__":
    main()
