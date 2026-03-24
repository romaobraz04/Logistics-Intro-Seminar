from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_opt.analysis.assignment5 import run_assignment5_suite
from battery_opt.config import load_assignment5_config, load_case_config


def main() -> None:
    case_config = load_case_config(PROJECT_ROOT / "configs" / "base_case.json")
    analysis_config = load_assignment5_config(PROJECT_ROOT / "configs" / "assignment5.json")
    output_dir = PROJECT_ROOT / "outputs" / "assignment5"
    run_assignment5_suite(case_config, analysis_config, output_dir)
    print(f"Wrote assignment 5 outputs to {output_dir}")


if __name__ == "__main__":
    main()
