from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pulp

from formulations.basic import Basic
from formulations.params import PARAMS
from formulations.tighter import Tighter

# assignment 7 runs with assignments 5 & 6 extensions
FEE_LEVELS = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
DEG_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.03]
FORMULATIONS = {
    "basic": Basic,
    "tighter": Tighter,
}
