
from simulation import Simulation
import datetime

train_start = datetime.datetime(2021, 1, 1, 0, 0, 0)
train_end = datetime.datetime(2022, 1, 4, 23, 0, 0)
test_start = datetime.datetime(2022, 1, 5, 0, 0,0)
test_end = datetime.datetime(2022, 1, 5, 23, 0, 0)

sim = Simulation(360, train_start, train_end, test_start, test_end)
forecasts = sim.run_simulation()
sim.plot_sim_results(forecasts)


