import sys
sys.path.insert(0, "C:\\Users\\jerom\\coding\\energy_demand_prediction_framework\\")

import datetime
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import statsmodels.api as sm
from data.quebec_energy_demand import HQ_data
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

from models.spline_regression import SplineRegression
from models.quadratic_regression import QuadraticRegression
from models.fourier_regression_simple import FourierRegressionSimple
from models.infinite_norm_min import InfiniteNormMinimization

class Combined_model:

    def get_predictions(self, data, train_start, train_end, test_start, test_end):

        pf = PolynomialFeatures(degree=1)
         
        spline_reg = SplineRegression()
        quad_reg = QuadraticRegression()
        fourier_reg = FourierRegressionSimple()
        inf_norm_reg = InfiniteNormMinimization()

        lin_reg_window_days = 15

        start_lin_reg = train_end - datetime.timedelta(days=lin_reg_window_days) + datetime.timedelta(hours=1)
        end_lin_reg = train_end

        spline_train_preds = spline_reg.get_predictions(data, train_start, train_end, start_lin_reg, end_lin_reg)
        quad_train_preds = quad_reg.get_predictions(data, train_start, train_end, start_lin_reg, end_lin_reg)
        inf_norm_reg_preds = inf_norm_reg.get_predictions(data, train_start, train_end, start_lin_reg, end_lin_reg)
        #fourier_train_preds = fourier_reg.get_predictions(data, train_start, train_end, start_lin_reg, end_lin_reg)
        

        model_fits = pd.DataFrame({
            "spline": spline_train_preds,
            "quad": quad_train_preds
        })
        model_fits = pf.fit_transform(model_fits)

        model = sm.OLS(data.loc[start_lin_reg:end_lin_reg, "demand"].values, model_fits).fit()
        
        # save model params
        params = model.params

        spline_preds = spline_reg.get_predictions(data, train_start, train_end, test_start, test_end)
        quad_preds = quad_reg.get_predictions(data, train_start, train_end, test_start, test_end)
        inf_norm_reg_preds = inf_norm_reg.get_predictions(data, train_start, train_end, test_start, test_end)
        #fourier_preds = fourier_reg.get_predictions(data, train_start, train_end, test_start, test_end)

        model_preds = pd.DataFrame({
            "spline": spline_preds,
            "quad": quad_preds
        })

        model_preds = pf.fit_transform(model_preds)

        forecast = model.predict(model_preds)

        return forecast, params



