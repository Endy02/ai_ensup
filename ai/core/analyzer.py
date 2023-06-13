import matplotlib.pyplot as plt  # for visualization
import seaborn as sns  # for coloring
from sklearn import metrics  # for evaluation
from sklearn import svm  # for Discriminator
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer  # To replace null value in dataframe
from sklearn.linear_model import SGDRegressor, BayesianRidge, ARDRegression, PassiveAggressiveRegressor, TheilSenRegressor, LinearRegression, LassoLars, HuberRegressor, Ridge   # For data analizis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split  # for fine-tuning
from sklearn.preprocessing import StandardScaler  # for feature scaling
import pandas as pd
import numpy as np
import json

from ai.tools.loader import Loader
from ai.tools.logger import Logger


class Analizer():
    def __init__(self):
        self.logger = Logger()
        self.loader = Loader()
        self.X, self.Y, self.W, self.X_train, self.X_test, self.Y_train, self.Y_test, self.W_train, self.W_test = self.__split_dataset(self.loader.get_final_dataset())

    def opitimization(self):
        """
            Optimized the selected model
        """
        print(" ---- Optimization ---- ")
        model = Ridge()
        model_params = {
            "alpha": [1,0.1,0.01,0.001,0.0001],
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
        model_gs = GridSearchCV(model, param_grid=model_params, n_jobs=1, verbose=3, error_score='raise')
        model_gs.fit(self.X_train, self.Y_train)

        opt_model = Ridge(random_state=3, **model_gs.best_params_)
        print(' ----- Optimized Model TEST ----- ')
        all_accuracies = cross_val_score(estimator=opt_model, X=self.X_train, y=self.Y_train, cv=10)
        print(all_accuracies)
        return opt_model

    def model_selection(self):
        """
            Model selection cross validation
        """
        try:
            models = [RandomForestRegressor(), Ridge(), HuberRegressor(max_iter=2000), BayesianRidge(), SGDRegressor(), LassoLars(), ARDRegression(), PassiveAggressiveRegressor(), TheilSenRegressor(), LinearRegression()]

            Y_scores = []
            W_scores = []

            for i, m in enumerate(models):
                df_fit = m.fit(self.X_train, self.Y_train)
                Y_scores.append({"type": type(m), "score": round(df_fit.score(self.X_test, self.Y_test) * 100, 2)})
            
            for i, m in enumerate(models):
                df_fit = m.fit(self.X_train, self.W_train)
                W_scores.append({"type": type(m), "score": round(df_fit.score(self.X_test, self.W_test) * 100, 2)})
            self.logger.info(f"TotalGHGEmissions Scores : {Y_scores}")
            self.logger.info(f"GHGEmissionsIntensity Scores : {W_scores}")
            
        except Exception as e:
            raise e
    
    def __split_dataset(self, df):
        try:
            X = df.drop(['TotalGHGEmissions', 'GHGEmissionsIntensity'], axis=1)
            X = df.drop(X.select_dtypes(include=['object']), axis=1)
            Y = df['TotalGHGEmissions']
            W = df['GHGEmissionsIntensity']

            # Standardization of X
            X = self.__standardization(X)

            # Split Data into taining and test sets
            X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(X, Y, W, train_size=0.7, random_state=0)
            self.logger.info(f"X_train Shape : {X_train.shape} -- Y_train Shape : {Y_train.shape} | X_test Shape : {X_test.shape} -- Y_test Shape: {Y_test.shape}, | W_train Shape : {W_train.shape} | W_test Shape : {W_test.shape}")
            
            return X, Y, W, X_train, X_test, Y_train, Y_test, W_train, W_test
        except Exception as e:
            raise e

    def __standardization(self, x):
        """
            Standardize train set 
        """
        try:
            scaler = StandardScaler()

            x_prep = x.select_dtypes(include=['float64','int64'])
            x_std = scaler.fit_transform(x_prep)
            x_std_df = pd.DataFrame(x_std, columns=x_prep.columns)

            return x_std_df
        except Exception as e:
            raise e