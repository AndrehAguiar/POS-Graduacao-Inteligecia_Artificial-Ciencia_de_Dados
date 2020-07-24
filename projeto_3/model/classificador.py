# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:39:20 2020

@author: TOP Artes
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

class Classificador(object):
    
    def __init__(self, df):
        
        df_class = df.query('MEDIA_NOTAS > 0').copy()

        self.X = df_class.drop(['ano', 'NM_UF_SIGLA', 'COD. MUNIC', 'NOME_MUNICÍPIO',
                                'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT',
                                'NU_NOTA_REDACAO','MEDIA_NOTAS','CLASSE'],
                               inplace=False, axis=1).values
        
        self.y = df_class['CLASSE'].values
    
    
    def scale_data(self, X):
        
        self.X_scaler = StandardScaler()  
        
        X_scaled = self.X_scaler.fit_transform(X)
    
        return X_scaled
    
        
    def select_features(self, model, X, y):
    
        sfm = SelectFromModel(model, threshold=0.04)
        sfm.fit(X, y.ravel())
        X = sfm.transform(X)
        
        return X
    
    
    def tree(self, X, base=True):
        """
        Adaboost + Decision Tree Classifier
        """

        kwargs = {'tree':{'random_state': 1, 'max_depth':3, 'criterion':'entropy'}, 'boost':{ 'n_estimators':1, 'learning_rate':1e-1}} if not base else {'tree':{},'boost':{}}
                
        X_train, X_test, y_train, y_test = train_test_split(X, self.y, random_state=1, test_size=0.35)
        
        classificador = AdaBoostClassifier(DecisionTreeClassifier(**kwargs['tree']),**kwargs['boost'])
        
        classificador.fit(X_train, y_train.ravel())
        
        score = classificador.score(X_test, y_test)
        
        print(f'Adaboost + Decision Tree Classifier - Score = {score}')
        
        previsoes = classificador.predict(X_test)
        
        return classificador
    
    
    def forest(self, X, base=True):
        """
        Random Forest Classifier
        """
        
        kwargs = {'random_state':1,'n_jobs':-1,'n_estimators':10, 'criterion':'gini', 'max_depth':1, 'max_features':'auto','ccp_alpha':1e-1,'min_impurity_decrease':1e-1} if not base else {}
        
        X_train, X_test, y_train, y_test = train_test_split(X, self.y, random_state=1, test_size=0.35)
        
        classificador = RandomForestClassifier(**kwargs)
        
        classificador.fit(X_train, y_train.ravel())
        
        score = classificador.score(X_test, y_test)
        
        print(f'Random Forest Classifier - Score = {score}')
        
        previsoes = classificador.predict(X_test)
        
        return classificador
    
    
    def svc(self, X, base=True):
        """
        SVC Support Vector Classifier
        """
        X_train, X_test, y_train, y_test = train_test_split(X, self.y, random_state=1, test_size=0.25)
        
        kwargs = {'max_iter':1500,'kernel':'rbf','tol':1e-3,'C':5.7,'decision_function_shape':'ovr', 'degree':1,'gamma':'scale','coef0':1, 'probability':True} if not base else {}
        
        classificador = SVC(**kwargs)
        
        classificador.fit(X_train, y_train.ravel())
        
        previsoes = classificador.predict(X_test)
        
        score = classificador.score(X_test, y_test)
        
        print(f'SVC Support Vector Classifier - Score = {score}')
        
        previsoes = classificador.predict(X_test)
            
        return classificador
    
    
    def mlp(self, X, base=True):

        """
        Neural Network - MLP Classifier
        """
        
        X_train, X_test, y_train, y_test = train_test_split(X, self.y, random_state=1, test_size=0.35)
        
        kwargs = {'max_iter':500,'early_stopping':True,'learning_rate_init':0.01,'activation':'tanh', 'solver':'adam','beta_1':0.5,'beta_2':0.1,'tol':1e-1, 'shuffle':True, 'learning_rate':'adaptive', 'hidden_layer_sizes':(int((X.shape[1])/2)+1,int((X.shape[1])/2)+1)} if not base else {}   
        
        classificador = MLPClassifier(**kwargs)
        
        classificador.fit(X_train, y_train.ravel())
        
        previsoes = classificador.predict(X_test)
        
        score = classificador.score(X_test, y_test)
        
        print(f'Neural Network - MLP Classifier - Score = {score}')
        
        previsoes = classificador.predict(X_test)
            
        return classificador
    
    
    def gradient(self, X, base=True):
        """
        Gradient Boosting 
        """
        
        X_train, X_test, y_train, y_test = train_test_split(X, self.y, random_state=1, test_size=0.35)
        
        kwargs = {'random_state':1,'criterion':'friedman_mse','max_features':'sqrt','loss':'deviance','learning_rate':0.01,'tol':1e-1, 'max_depth':3, 'n_estimators':3} if not base else {}   
        
        classificador = GradientBoostingClassifier(**kwargs)
        
        classificador.fit(X_train, y_train.ravel())
        
        previsoes = classificador.predict(X_test)
        
        score = classificador.score(X_test, y_test)
        
        print(f'Gradient Boosting - Score = {score}')
        
        previsoes = classificador.predict(X_test)
            
        return classificador
    
    def voting(self, X, y, models):
        """
        Voting Classifier
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.35)
        
        ecls = VotingClassifier(estimators=models, voting='soft', flatten_transform=True)
        
        ecls.fit(X_train, y_train.ravel())
        
        previsoes = ecls.predict(X_test)
        
        score = ecls.score(X_test, y_test)
        
        print(score)
        
        previsoes = ecls.predict(X_test)
        
        return ecls