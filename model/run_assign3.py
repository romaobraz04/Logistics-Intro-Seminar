import pandas as pd
import pulp
from formulations.params import *
from formulations.basic import Basic
from formulations.tighter import Tighter


data = pd.read_csv("data/net_demand_and_price.csv")

print("Running Basic formulation...")
basic_model = pulp.LpProblem("Battery_Optimization_Basic", pulp.LpMaximize)
basic = Basic(basic_model, data, PARAMS)
basic.run_model()
print("\nRunning Tighter formulation...")
tighter_model = pulp.LpProblem("Battery_Optimization_Tighter", pulp.LpMaximize)
tighter = Tighter(tighter_model, data, PARAMS)
tighter.run_model()
