import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (GradientBoostingRegressor, 
                              RandomForestRegressor,
                              AdaBoostRegressor)
from sklearn.svm import SVR, SVC
import sklearn.model_selection as cv
from sklearn.metrics import mean_squared_error, log_loss, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def get_filepath(config):

    filepath = config.filepath
    df = pd.read_csv(filepath)
    X_columns = ['posts', 'followed_by', 'year', 'month', 'day_of_week', 'hour','type_GraphSidecar',
             'type_GraphVideo', 'l__1', 'l__2', 'l__3', 'l__4', 'l__5', 'l__6', 'l__7',
             'tagged_users', 'caption_words', 'caption_hashtags','days_since_last_post', 
             'aesthetic', 'std_aesthetic', 'technical','std_technical', 'x_predict', 'vgg19_predict', 'engagement_ratio']
    df = df[X_columns].copy()
    return df

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators = 2000, n_jobs = -1)
knn = KNeighborsRegressor()

models = [lr, rf, knn]
model_names = ['lr', 'rf', 'knn']



def get_train_test_split(df):
    """Get numerical columns from df, return X and y"""
    X = df.drop('engagement_ratio', axis = 1).copy()
    Y = df['engagement_ratio']
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
    return X_train, X_test, y_train, y_test

def run_model(models, X_train, X_test, y_train, y_test):
    for model, model_name in zip(models, model_names):
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        mse = mean_squared_error(y_test, yhat)
        mae = mean_absolute_error(y_test, yhat)
        r2 = r2_score(y_test, yhat)
        rmse = np.sqrt(mse)
        print(f'{model_name} r2: {r2:.3f}, mae: {mae:.3f}, mse: {mse:.3f}, rmse: {rmse:.3f}')
        
    A = df.drop('engagement_ratio', axis = 1).copy()
    b = df['engagement_ratio']
    A = np.column_stack([np.ones(A.shape[0]), A])
    
    X_train, X_test, y_train, y_test = train_test_split(A, b, test_size=0.20, random_state=42)
    # calculate the economy SVD for the data matrix A
    U,S,Vt = np.linalg.svd(X_train, full_matrices=False)
    # solve Ax = b for the best possible approximate solution in terms of least squares
    x_hat = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y_train
    # perform train and test inference
    y_pred = X_train @ x_hat
    yhat = X_test @ x_hat
    mse = mean_squared_error(y_test, yhat)
    mae = mean_absolute_error(y_test, yhat)
    r2 = r2_score(y_test, yhat)
    rmse = np.sqrt(mse)
    print(f'svd r2: {r2:.3f}, mae: {mae:.3f}, mse: {mse:.3f}, rmse: {rmse:.3f}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--filepath', type=str)

    config = parser.parse_args()

    df = get_filepath(config)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    run_model(models, X_train, X_test, y_train, y_test)