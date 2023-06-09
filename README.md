# energy_demand_prediction_framework
Framework to test forecast models on Hydro-Quebec energy consumption data.

## How to run simulation:

1. In run_simulation file, first input initial train_start, train_end, test_start and test_end dates on top. Then create Simulation object with above quantities as well as the number of days to simulate. EX: sim = Simulation(3, train_start, train_end, test_start, test_end).

2. In simulation file, implement get_prediction function to return a forecast of 24 time steps.

3. Results are saved in results folder.

