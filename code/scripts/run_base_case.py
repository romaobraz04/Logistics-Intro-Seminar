from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_opt.config import Formulation, load_case_config
from battery_opt.solve import solve_case


def main() -> None:
    config = load_case_config(PROJECT_ROOT / "configs" / "base_case.json")
    for formulation in (Formulation.BASIC, Formulation.TIGHTER):
        result = solve_case(config, formulation)
        print(
            f"{formulation.value}: "
            f"status={result.status_name}, "
            f"objective_eur={result.objective_value_eur}, "
            f"runtime_seconds={result.runtime_seconds:.4f}"
        )


if __name__ == "__main__":
    main()
