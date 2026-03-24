from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_opt.analysis.assignment6 import run_assignment6_suite
from battery_opt.config import load_assignment6_config, load_case_config


def main() -> None:
    case_config = load_case_config(PROJECT_ROOT / "configs" / "base_case.json")
    analysis_config = load_assignment6_config(PROJECT_ROOT / "configs" / "assignment6.json")
    output_dir = PROJECT_ROOT / "outputs" / "assignment6"
    run_assignment6_suite(case_config, analysis_config, output_dir)
    print(f"Wrote assignment 6 outputs to {output_dir}")


if __name__ == "__main__":
    main()
