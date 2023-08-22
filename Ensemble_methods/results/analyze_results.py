import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results = pd.read_csv("Ensemble_methods\\results\\stacked_model_ridge.csv")
ridge_parameters = pd.read_csv("Ensemble_methods\\results\\stacked_model_ridge_parameters.csv")
ols_parameters = pd.read_csv("Ensemble_methods\\results\\stacked_model_ols_parameters.csv")


# models

ridge_rmse = np.sqrt(np.mean(np.square(results.loc[:, "demand"] - results.loc[:, "forecast"])))
spline_rmse = np.sqrt(np.mean(np.square(results.loc[:, "demand"] - results.loc[:, "spline"])))
fourier_rmse = np.sqrt(np.mean(np.square(results.loc[:, "demand"] - results.loc[:, "fourier"])))
sarimax_rmse = np.sqrt(np.mean(np.square(results.loc[:, "demand"] - results.loc[:, "sarimax"])))
mlp_rmse = np.sqrt(np.mean(np.square(results.loc[:, "demand"] - results.loc[:, "mlp"])))
ols_rmse= 601

rmses = [mlp_rmse, sarimax_rmse, spline_rmse, fourier_rmse, ols_rmse, ridge_rmse]
plt.bar(["MLP", "Sarimax", "Regression splines", "Fourier", "OLS", "Ridge"], rmses, width=0.7)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("RMSE of individual models")
plt.show()

ridge_mape = np.mean(np.abs((results.loc[:, "demand"] - results.loc[:, "forecast"]) / results.loc[:, "demand"])) * 100
spline_mape = np.mean(np.abs((results.loc[:, "demand"] - results.loc[:, "spline"]) / results.loc[:, "demand"])) * 100
fourier_mape = np.mean(np.abs((results.loc[:, "demand"] - results.loc[:, "fourier"]) / results.loc[:, "demand"])) * 100
sarimax_mape = np.mean(np.abs((results.loc[:, "demand"] - results.loc[:, "sarimax"]) / results.loc[:, "demand"])) * 100
mlp_mape = np.mean(np.abs((results.loc[:, "demand"] - results.loc[:, "mlp"]) / results.loc[:, "demand"])) * 100
ols_mape = 1.91

mapes = [sarimax_mape, spline_mape, fourier_mape, ols_mape, ridge_mape]   
plt.bar(["Sarimax", "Regression splines", "Fourier", "OLS", "Ridge"], mapes, width=0.7)
plt.xlabel("Model")
plt.ylabel("MAPE")
plt.title("MAPE of individual models and ensemble")
plt.show()


# plot results
plt.plot(ridge_parameters.loc[:, "spline"], label="Ridge")
plt.plot(ols_parameters.loc[:, "spline"], label="OLS")
plt.xlabel("Time")
plt.ylabel("Parameter value")
plt.title("Evolution of weight of regression spline model in ensemble")
plt.legend()
plt.show()