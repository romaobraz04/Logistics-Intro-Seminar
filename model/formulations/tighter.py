from .common import BatteryOptimizer


class Tighter(BatteryOptimizer):
    """Tighter formulation of battery optimization problem with improved constraints."""

    def add_soc_constraints(self):
        """Add tighter state of charge constraints using previous period state."""
        self.model += (
            self.dec_vars[f"soc_{self.periods}"] >= self.min_soc,
            f"Minimum SOC_{self.periods}",
        )
        self.model += (
            self.dec_vars[f"soc_{self.periods}"] <= self.max_soc,
            f"Maximum SOC_{self.periods}",
        )
        for t in range(self.periods):
            self.model += (
                self.dec_vars[f"soc_{t}"]
                >= self.min_soc
                + self.delta
                * self.dec_vars[f"discharge_power_{t}"]
                / self.discharge_efficiency,
                f"Minimum SOC_{t}",
            )
            self.model += (
                self.dec_vars[f"soc_{t}"]
                <= self.max_soc
                - self.delta
                * self.dec_vars[f"charge_power_{t}"]
                * self.charge_efficiency,
                f"Maximum SOC_{t}",
            )
