import pulp
import pandas as pd


class BatteryOptimizer:
    """Base class for battery optimization models."""

    def __init__(
        self,
        model,
        data,
        params,
        lp=False,
        grid_fee_eur_kwh=0.0,
        degradation_cost_eur_per_kwh_discharge=0.0,
        msg=True,
    ):
        self.model = model
        self.data = data
        self.params = params
        self.lp_relaxation = lp
        self.grid_fee_eur_kwh = grid_fee_eur_kwh
        self.degradation_cost_eur_per_kwh_discharge = (
            degradation_cost_eur_per_kwh_discharge
        )
        self.msg = msg
        # read data
        self.net_demand = data["Volume (kWh)"].reset_index(drop=True)
        self.price = data["Price (EUR/kWh)"].reset_index(drop=True)
        self.delta = 1.0  # time step in hours
        self.periods = len(self.net_demand)  # number of time periods

        self.dec_vars = {}  # dictionary to hold decision variables

        # parameters
        self.init_soc = params["init_soc"]
        self.min_soc = params["min_soc"]
        self.max_soc = params["max_soc"]
        self.charge_limit = params["charge_limit"]
        self.discharge_limit = params["discharge_limit"]
        self.charge_efficiency = params["charge_efficiency"]
        self.discharge_efficiency = params["discharge_efficiency"]

    # create decision variables
    def create_variables(self):
        for t in range(self.periods + 1):
            self.dec_vars[f"soc_{t}"] = pulp.LpVariable(
                f"State_of_Charge_{t}", lowBound=0, cat="Continuous"
            )
        for t in range(self.periods):
            self.dec_vars[f"electricity_buy_{t}"] = pulp.LpVariable(
                f"Electricity_Bought_{t}", lowBound=0, cat="Continuous"
            )
            self.dec_vars[f"electricity_sell_{t}"] = pulp.LpVariable(
                f"Electricity_Sold_{t}", lowBound=0, cat="Continuous"
            )
            self.dec_vars[f"charge_power_{t}"] = pulp.LpVariable(
                f"Charging_Power_{t}", lowBound=0, cat="Continuous"
            )
            self.dec_vars[f"discharge_power_{t}"] = pulp.LpVariable(
                f"Discharging_Power_{t}", lowBound=0, cat="Continuous"
            )
            if self.lp_relaxation:
                self.dec_vars[f"mode_{t}"] = pulp.LpVariable(
                    f"Mode_Indicator_{t}", lowBound=0, upBound=1, cat="Continuous"
                )
            else:
                self.dec_vars[f"mode_{t}"] = pulp.LpVariable(
                    f"Mode_Indicator_{t}", cat="Binary"
                )
        print("Variables created.") if self.msg else None

    # objective function
    def objective_function(self):
        energy_profit = pulp.lpSum(
            self.delta
            * self.price[t]
            * (
                self.dec_vars[f"electricity_sell_{t}"]
                - self.dec_vars[f"electricity_buy_{t}"]
            )
            for t in range(self.periods)
        )

        fee_penalty = 0
        if self.grid_fee_eur_kwh > 0.0:
            fee_penalty = pulp.lpSum(
                self.delta
                * self.grid_fee_eur_kwh
                * (
                    self.dec_vars[f"electricity_buy_{t}"]
                    + self.dec_vars[f"electricity_sell_{t}"]
                )
                for t in range(self.periods)
            )

        degradation_penalty = 0
        if self.degradation_cost_eur_per_kwh_discharge > 0.0:
            degradation_penalty = pulp.lpSum(
                self.delta
                * self.degradation_cost_eur_per_kwh_discharge
                * self.dec_vars[f"discharge_power_{t}"]
                for t in range(self.periods)
            )

        self.model += energy_profit - fee_penalty - degradation_penalty
        print("Objective function defined.") if self.msg else None

    # constraints - to be implemented by subclasses
    def add_soc_constraints(self):
        """Add state of charge constraints. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement add_soc_constraints()")

    def add_constraints(self):
        # Initial SOC constraint
        self.model += (
            self.dec_vars["soc_0"] == self.init_soc,
            "Initial SOC",
        )

        # SOC update constraints
        for t in range(self.periods):
            self.model += (
                self.dec_vars[f"soc_{t+1}"]
                == self.dec_vars[f"soc_{t}"]
                + (
                    self.dec_vars[f"charge_power_{t}"] * self.charge_efficiency
                    - self.dec_vars[f"discharge_power_{t}"] / self.discharge_efficiency
                )
                * self.delta,
                f"SOC Update_{t+1}",
            )

        # Add SOC min/max constraints (implementation depends on subclass)
        self.add_soc_constraints()

        # Energy balance and mode constraints
        for t in range(self.periods):
            self.model += (
                self.delta
                * (
                    self.dec_vars[f"electricity_buy_{t}"]
                    - self.dec_vars[f"electricity_sell_{t}"]
                    - self.dec_vars[f"charge_power_{t}"]
                    + self.dec_vars[f"discharge_power_{t}"]
                )
                == self.net_demand.iloc[t],
                f"Energy Balance_{t}",
            )
            self.model += (
                self.dec_vars[f"charge_power_{t}"]
                <= self.charge_limit * self.dec_vars[f"mode_{t}"],
                f"Charge Limit_{t}",
            )
            self.model += (
                self.dec_vars[f"discharge_power_{t}"]
                <= self.discharge_limit * (1 - self.dec_vars[f"mode_{t}"]),
                f"Discharge Limit_{t}",
            )
        print("Constraints added.") if self.msg else None

    # run model
    def run_model(self):
        self.create_variables()
        self.objective_function()
        self.add_constraints()
        status = self.model.solve(pulp.HiGHS(msg=False))
        print(f"Status: {pulp.LpStatus[status]}") if self.msg else None
        (
            print(f"Optimal Value: {pulp.value(self.model.objective)}")
            if self.msg
            else None
        )
        print(f"Runtime: {self.model.solutionTime}s") if self.msg else None

        # Extract decision variable values into a dataframe
        vars = {}
        for var_name, var in self.dec_vars.items():
            vars[var_name] = var.varValue

        self.vars_df = pd.DataFrame(vars, index=[0])
        return (
            status,
            pulp.value(self.model.objective),
            self.model.solutionTime,
            self.vars_df,
        )
