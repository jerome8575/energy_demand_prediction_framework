
from simulation import Simulation
import datetime

train_start = datetime.datetime(2019, 1, 2, 0, 0, 0)
train_end = datetime.datetime(2020, 12, 31, 23, 0, 0)
test_start = datetime.datetime(2021, 1, 1, 0, 0,0)
test_end = datetime.datetime(2021, 1, 1, 23, 0, 0)

sim = Simulation(360, train_start, train_end, test_start, test_end)
forecasts, params = sim.run_simulation()
params.to_csv("params.csv")
sim.plot_sim_results(forecasts)


