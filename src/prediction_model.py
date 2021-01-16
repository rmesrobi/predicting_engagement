import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (GradientBoostingRegressor, 
                              RandomForestRegressor,
                              AdaBoostRegressor)
from sklearn.svm import SVR, SVC
import sklearn.model_selection as cv
from sklearn.metrics import mean_squared_error, log_loss, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('~/Desktop/galvanize/capstone_3/external_df.csv')

def get_train_test_split(self):
    """Get numerical columns from df, return X and y"""
    df_numeric = self.df.select_dtypes(include = np.number)

    X_columns = df_numeric.columns.drop('engagement_ratio')
    X = df_numeric[X_columns]
    Y = df_numeric['engagement_ratio']
    X_train, X_test, y_train, y_test = train_test_split(X,Y)
    return X_train, X_test, y_train, y_test
