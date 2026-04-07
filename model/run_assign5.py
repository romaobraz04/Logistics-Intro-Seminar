import pandas as pd

from helpers.assign5.wrapper import assign5wrapper

data = pd.read_csv("data/net_demand_and_price.csv")

if __name__ == "__main__":
    assign5 = assign5wrapper(data)
    assign5.run()
