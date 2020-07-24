# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:53:39 2020

@author: TOP Artes
"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel


class Regressor(object):
    
    def __init__(self, df):
        
        df_reg = df.query('MEDIA_NOTAS > 0').copy()
        
        self.X = df_reg.drop(['ano', 'NM_UF_SIGLA', 'COD. MUNIC', 'NOME_MUNICÍPIO',
             'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT',
             'NU_NOTA_REDACAO','MEDIA_NOTAS'], inplace=False, axis=1).values
        
        self.y = df.query('MEDIA_NOTAS > 0')['MEDIA_NOTAS'].values
    
    

    def scale_data(self, X, y):
        
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()   
        
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1,1))
    
        return X_scaled, y_scaled
    
    
    
    def select_features(self, model, X, y):
    
        sfm = SelectFromModel(model, threshold=0.004)
        sfm.fit(X, y.ravel())
        X = sfm.transform(X)     
        
        return X
    
    
    
    def get_params(self, X, y, model, grid):
        
        random = RandomizedSearchCV(scoring="neg_mean_absolute_error", estimator = model, param_distributions = grid, n_iter = 100, cv = 3, verbose=2, random_state=1, n_jobs = -1)
            # Fit the random search model
        random.fit(X, y.ravel())
        
        return random.best_params_
    
    
    
    def tree(self, X, y, base=True, plot=False):
        """
        Adaboost + Decision Tree Regressors
        """
        def get_grid():
            
            grid = {'max_depth':[110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300,  None],
                    'max_leaf_nodes':[2, 5, 10, 15, 20, None],
                    'min_impurity_decrease':[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                    'max_features':['auto','sqrt'],
                    'min_samples_leaf': [1, 2, 4],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'criterion':['friedman_mse', 'mse']}
            
            return grid
        
            
        kwargs = {'random_state':1, 'loss':'square', 'n_estimators':500, 'learning_rate':1e-6} if not base else {'random_state':1}
        
        
        regressor = AdaBoostRegressor(DecisionTreeRegressor(random_state= 1),**kwargs) if base else AdaBoostRegressor(DecisionTreeRegressor(random_state= 1, **self.get_params(X, y, DecisionTreeRegressor(), get_grid())),**kwargs)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.35)
        
        regressor.fit(X_train, y_train.ravel())
        
        previsoes = self.y_scaler.inverse_transform(regressor.predict(X_test))
        y_test = self.y_scaler.inverse_transform(y_test)
        
        mae = mean_absolute_error(y_test, previsoes)
        
        print(f'{regressor} - MAE = {mae}\n\n')
        
        if plot:
            self.plot_results(y_test, previsoes, 'Decision Tree', mae)
        
        return regressor
    


    def forest(self, X, y, base=True, plot=False):
        """
        Random Forest Regressor
        """        
        def get_grid():
            
            grid = {'bootstrap': [True, False],
               'max_depth': [110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300,  None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10, 15, 20],
               'n_estimators': [130, 180, 230, 280, 320, 380, 420, 470, 520],
               'ccp_alpha':[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
               'min_impurity_decrease':[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]}
            
            return grid
        
        regressor = RandomForestRegressor(random_state=1) if base else RandomForestRegressor(random_state=1, **self.get_params(X, y, RandomForestRegressor(), get_grid()))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.35)
        
        regressor.fit(X_train, y_train.ravel())
        
        previsoes = self.y_scaler.inverse_transform(regressor.predict(X_test))
        y_test = self.y_scaler.inverse_transform(y_test)
        
        mae = mean_absolute_error(y_test, previsoes)
        
        print(f'{regressor} - MAE = {mae}\n\n')
        
        if plot:
            self.plot_results(y_test, previsoes, 'Random Forest', mae)

        return regressor
    
    
    
    def svr(self, X, y, base=True, plot=False):
        """
        SVR Support Vector Regressor
        """
        
        def get_grid():
            
            grid = {'max_iter':[200,500,1000,2000,3000,4000,5000],  
                    'kernel':['rbf'],
                    'tol':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],
                    'C':[1,3,5,7,11,13,17,19,23,27,31,37,41,45],
                    'epsilon':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                    'degree':[1,2,3,4,5],
                    'gamma':['scale','auto'],
                    'coef0':[1],
                    'shrinking':[False,True]}
            
            return grid
        
        regressor = SVR() if base else SVR(**self.get_params(X, y, SVR(), get_grid()))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.35)
        
        regressor.fit(X_train, y_train.ravel())
        
        previsoes = self.y_scaler.inverse_transform(regressor.predict(X_test))
        y_test = self.y_scaler.inverse_transform(y_test)
        
        mae = mean_absolute_error(y_test, previsoes) 
        
        print(f'{regressor} - MAE = {mae}\n\n')
        
        if plot:
            self.plot_results(y_test, previsoes, 'Support Vector', mae)
            
        return regressor
    
    
    
    def mlp(self, X, y, base=True, plot=False):

        """
        Neural Network - MLP Regressor
        """
        
        def get_grid():
            
            grid = {'max_iter':[100,200,300,400,500,600,700,800,900,1000],
                    'early_stopping':[False,True],
                    'momentum':[0.1,0.3,0.5,0.9],
                    'nesterovs_momentum':[True, False],
                    'validation_fraction':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],
                    'learning_rate_init':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],
                    'activation':['identity','logistic','tanh','relu'],
                    'solver':['sgd','lbfgs','adam'],
                    'beta_1':[0.1,0.3,0.5,0.9],
                    'beta_2':[0.991,0.993,0.995,0.999],
                    'tol':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],
                    'alpha':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],
                    'shuffle':[True],
                    'learning_rate':['constant','invscaling','adaptive'],
                    #'power_t':[1,3,5,7,9,11,13,15,17,19,23,27,29,33,37,41,45],
                    'warm_start':[True,False]}
            
            return grid
        
        
        regressor = MLPRegressor(random_state=1) if base else MLPRegressor(random_state=1, **self.get_params(X, y, MLPRegressor(), get_grid()), hidden_layer_sizes=(int((X.shape[1])/2)+1,int((X.shape[1])/2)+1))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.35)
        
        regressor.fit(X_train, y_train.ravel())
        
        previsoes = self.y_scaler.inverse_transform(regressor.predict(X_test))
        y_test = self.y_scaler.inverse_transform(y_test)
        
        mae = mean_absolute_error(y_test,previsoes)
                
        print(f'{regressor} - MAE = {mae}\n\n')
        
        if plot:
            self.plot_results(y_test, previsoes,'Neural Network', mae)
        
        return regressor
    


    def gradient(self, X, y, base=True, plot=False, estimator=None):
        """
        Gradient Boosting 
        """
        
        def get_grid():
            
            grid = {'criterion':['mse','friedman_mse'],
                    'max_features':['auto','sqrt'],
                    'loss':['ls','lad','huber','quantile'],
                    'learning_rate':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],
                    'subsample':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],
                    'tol':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],
                    'max_depth':[1,3,5,7,9,11,13,15,17,19,23,25,29,35],
                    'min_samples_split':[1,3,5,7,9,11],
                    'n_estimators':[100,200,300,400,500],
                    #'validation_fraction':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],
                    'warm_start':[True, False],
                    'alpha':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],
                    'ccp_alpha':[0.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]}
            
            return grid
        
        regressor = GradientBoostingRegressor(random_state=1) if base else GradientBoostingRegressor(random_state=1, **self.get_params(X, y, GradientBoostingRegressor(), get_grid()))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.35)  
        
        regressor.fit(X_train, y_train.ravel())
        
        previsoes = self.y_scaler.inverse_transform(regressor.predict(X_test))
        y_test = self.y_scaler.inverse_transform(y_test)
        
        mae = mean_absolute_error(y_test, previsoes)
        
        print(f'{regressor} - MAE = {mae}\n\n')
        
        if plot:
            self.plot_results(y_test, previsoes, 'Gradient Boosting', mae)
            
        return regressor
    
    

    def voting(self, X, y, models, select=False):
        """
        Voting Regressor
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.35)
        
        regressor = VotingRegressor(models)
        
        regressor.fit(X_train, y_train.ravel())
        
        previsoes = self.y_scaler.inverse_transform(regressor.predict(X_test))
        y_test = self.y_scaler.inverse_transform(y_test)
        
        mae = mean_absolute_error(y_test, previsoes)
        
        self.plot_results(y_test, previsoes, 'Voting Regressor', mae, select)
        
        return regressor
    
    
    
    def plot_results(self, y, previsoes, model, mae, select):
        
        plt.figure(figsize=(20,20))
        plt.subplot(2,2,1)
        ax = sns.regplot(y, previsoes, marker='o', color='c', scatter_kws={'s':20,'edgecolor':'w',"alpha":0.7}, label='Targets Vs Previsões')
        
        if select:
            plt.scatter(y, y, s=20, label='Orginais')
            
        ax.set(xlim=(0,1000), ylim=(0,1000), xlabel='Targets', ylabel='Previsões', title=f'Targets Vs. Previsões\n{model} (MAE = {round(mae, 4)})')
        plt.legend()
        
        plt.subplot(2,2,2)
        ax = sns.regplot(y, previsoes-y.ravel(), marker='o', color='r', scatter_kws={'s':20,'edgecolor':'w',"alpha":0.7}, label='Diferença (Previsões - Targets)')
        ax.set(xlim=(0,1000), ylim=(-500,500), xlabel='Targets', ylabel='Previsões - Targets', title=f'Diferença Previsões - Targets\n{model} (Média dos erros = {round((previsoes-y.ravel()).mean(), 4)})')
        
        return