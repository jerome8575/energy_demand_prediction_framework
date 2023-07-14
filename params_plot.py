import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

params = pd.read_csv("params.csv")
params = params.drop(columns=["Unnamed: 0"])
params = params.drop(columns=["Intercept"])
print(params)

plt.plot(params)
plt.legend(params.columns)
plt.show()