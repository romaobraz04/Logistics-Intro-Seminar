from .common import BatteryOptimizer


class Basic(BatteryOptimizer):
    """Basic formulation of battery optimization problem."""

    def add_soc_constraints(self):
        """Add simple state of charge constraints for each period."""
        for t in range(self.periods + 1):
            self.model += (
                self.dec_vars[f"soc_{t}"] >= self.min_soc,
                f"Minimum SOC_{t}",
            )
            self.model += (
                self.dec_vars[f"soc_{t}"] <= self.max_soc,
                f"Maximum SOC_{t}",
            )
