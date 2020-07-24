# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:08:15 2020

@author: TOP Artes
"""

# Importa a biblioteca para estruturação dos dados
import numpy as np

# Importa as classes necessárias
from model.regressor import Regressor
from model.validator import Validator

from control.control_validator import ControlValidator
from control.control_analise import ControlAnalise

# Importa a biblioteca para separação dos dados em Treino e Teste
from sklearn.model_selection import train_test_split


class ControlRegressor(object):
    
    def __init__(self):
        
        self.regressor          = Regressor()
        self.validator          = Validator()
        self.control_analise    = ControlAnalise()
        self.control_validator  = ControlValidator()
        self.lst_modelo         = ['arvore', 'linear_poly','rede_neural','support_vector']
        
    
    def set_kwargs(self, base, baseline): 
        
        kwargs = []
                 
        if len(baseline) > 0:
            base = False
            
        kwargs.append({'tree':{'max_depth':15, 'criterion':'friedman_mse'},'boost':{'n_estimators':1000, 'learning_rate':.00001},'tunned':{'test':'0'}}) if base is False else kwargs.append({'tree':{}})
        kwargs.append({'fit_intercept':False, 'n_jobs':-1, 'normalize':True}) if base is False else kwargs.append({})
        kwargs.append({'max_iter':3000,'early_stopping':True, 'activation':'tanh', 'solver':'lbfgs','alpha':0.0005,'tol':1e-5, 'shuffle':True, 'learning_rate':'adaptive'}) if base is False else kwargs.append({})
        kwargs.append({'max_iter':10000,'kernel':'rbf','tol':7e-6,'C':3.7,'epsilon':0.03, 'degree':1,'gamma':'scale','coef0':0}) if base is False else kwargs.append({})
        
        for dct in list(baseline):
            if dct in list(baseline):
                kwargs[dct] = {'tree':{}} if dct == 0 else {}
                
        return kwargs 
    
    
    
    #len('boost' in {'tree':{'max_depth':15, 'criterion':'friedman_mse'},'boost':{'n_estimators':250, 'learning_rate':.1},'tunned':{}})

    def train_test(self, X, y, test_size, random_state):
        
        # Divide a base em treino e teste com valor de 15% para teste (BASE DE DADOS PEQUENA)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return X_train, X_test, y_train, y_test
        
    
        
    def get_metrics(self, X, y, pre, random_state, base, scaled, plt_text, config, kwargs, validation, plot, modelo=None, selection=False) -> np.array:
             
        dct_text = None
        
        def get_args(X, y, i, pre, config, scaled, kwargs):
            
            pre                              = pre if i not in config else config[i]['pre']
            kwargs                           = kwargs[i] if i not in config else config[i]['kwargs'][i]
            scaled                           = scaled if i not in config else config[i]['scaled']
            X                                = X if i not in config else config[i]['X']
            y                                = y if i not in config else config[i]['y']     
            
            return X, y, pre, kwargs, scaled 
        
            
        if plot:
            dct_text = self.control_analise.set_plt_text(plt_text)
            
        if modelo==None or modelo =='arvore': 
            
            X_ad, y_ad, pre_ad, kwargs_ad, scaled_ad = get_args(X, y, 0, pre, config, scaled, kwargs) 
            
            X_ad        = (pre_ad['scaler'][0][0], pre_ad['scaler'][0][1], X_ad[2]) if scaled_ad else X_ad
            y_ad        = pre_ad['scaler'][1] if scaled_ad else y_ad         
            
            kwargs_ad['tree']['random_state']   = random_state            
                
            
            if modelo =='arvore' and validation:
                X, y = np.vstack((X_ad[0], X_ad[1])), np.vstack((y_ad[0], y_ad[1]))
                
                # Retorna array com os 30 resultados
                return self.regressor.treinar_arvore(X[:,1:], y, pre_ad, scaled_ad, plot, dct_text, validation, kwargs_ad)
            
            # Invoca a função de treinamento do modelo passando os hiperparâmetros e valores de treino(X) e teste(y)
            # Regressão com Árvore de Decisão
            feat_importances_arvore, score_arvore, mae_arvore, modelo_arvore =  self.regressor.treinar_arvore(
                X_ad, y_ad, pre_ad, scaled_ad, plot, dct_text, validation, kwargs_ad)            
            
            if modelo!=None and selection is True:
                return (pre['scaler'][2],pre['scaler'][3], feat_importances_arvore, score_arvore, mae_arvore, modelo_arvore)
                  
            
            
        
        if modelo==None or modelo =='linear_poly':
            
            X_lp, y_lp, pre_lp, kwargs_lp, scaled_lp = get_args(X, y, 1, pre, config, scaled, kwargs)
            
            X_lp            = (pre_lp['poly'][1].transform(pre_lp['scaler'][0][0]), #pre_lp['poly'][0][0][:,1:]
                               pre_lp['poly'][1].transform(pre_lp['scaler'][0][1]), #pre_lp['poly'][0][1][:,1:]
                               X_lp[2]) if scaled_lp else (pre_lp['poly'][0][0], pre_lp['poly'][0][1], X_lp[2])
            y_lp            = pre_lp['scaler'][1] if scaled_lp else y_lp
            
            if modelo =='linear_poly' and  validation:
                
                X, y = np.vstack((X_lp[0], X_lp[1])), np.vstack((y_lp[0], y_lp[1]))
                
                # Retorna array com os 30 resultados
                return self.regressor.treinar_linear_poly(X, y, pre_lp, scaled_lp, plot, dct_text, validation, kwargs_lp) 
            
            # Invoca a função de treinamento do modelo passando os hiperparâmetros e valores de treino(X) e teste(y)
            # Regressão Linear(Polinomial)
            
            score_poly, mae_poly, modelo_poly = self.regressor.treinar_linear_poly(
                X_lp, y_lp, pre_lp, scaled_lp, plot, dct_text, validation, kwargs_lp)
            
            if modelo!=None and selection is True:
                return (pre['scaler'][2],pre['scaler'][3], score_poly, mae_poly, modelo_poly)
            
            
     
        if modelo==None or modelo =='rede_neural':
            
            X_rn, y_rn, pre_rn, kwargs_rn, scaled_rn = get_args(X, y, 2, pre, config, scaled, kwargs)
            
            X_rn            = (pre_rn['scaler'][0][0], pre_rn['scaler'][0][1], X_rn[2]) 
            y_rn            = pre_rn['scaler'][1]
                
            node = int((X_rn[2].shape[1])/2)+1
            
            kwargs_rn['hidden_layer_sizes'] = (node,node)
            # kwargs_rn['random_state']       = random_state 
            
            # Define a quantidade de camadas ocultas
            # Invoca a função de treinamento do modelo passando os hiperparâmetros e valores de treino(X) e teste(y)
            # Regressão com Rede neural(MPL) 
            if modelo =='rede_neural' and  validation is True:
                
                X, y = np.vstack((X_rn[0], X_rn[1])), np.vstack((y_rn[0], y_rn[1]))
                
                # Retorna array com os 30 resultados
                return self.regressor.treinar_redeneural(X[:,1:], y, pre_rn, scaled_rn, plot, dct_text, validation, kwargs_rn)
        
            score_mpl, mae_mpl, modelo_mpl = self.regressor.treinar_redeneural(
                X_rn, y_rn, pre_rn, scaled_rn, plot, dct_text, validation, kwargs_rn)
                
            if modelo!=None and selection is True:
                return (pre['scaler'][2],pre['scaler'][3], score_mpl, mae_mpl, modelo_mpl)
                 
            
            
            
        if modelo==None or modelo == 'support_vector':
            
            X_sv, y_sv, pre_sv, kwargs_sv, scaled_sv = get_args(X, y, 3, pre, config, scaled, kwargs)
            
            X_sv            = (pre_sv['scaler'][0][0], pre_sv['scaler'][0][1], X_sv[2])            
            y_sv            = pre_sv['scaler'][1]
                
            if modelo == 'support_vector' and  validation is True:
                X, y = np.vstack((X_sv[0],X_sv[1])), np.vstack((y_sv[0], y_sv[1]))
                # Retorna array com os 30 resultados
                return self.regressor.treinar_svr(X, y, pre_sv, scaled_sv, plot, dct_text, validation, kwargs_sv)            
            
            # Invoca a função de treinamento do modelo passando os hiperparâmetros e valores de treino(X) e teste(y)
            # Regressão Linear(Polinomial)
            score_svr, mae_svr, modelo_svr = self.regressor.treinar_svr(
                X_sv, y_sv, pre_sv, scaled_sv, plot, dct_text, validation, kwargs_sv)
            
            if modelo!=None and selection is True:
                return (pre['scaler'][2], pre['scaler'][3], score_svr, mae_svr, modelo_svr) 
        
        dict_base_line = {'arvore':(score_arvore, mae_arvore),
                          'linear_poly':(score_poly, mae_poly),
                          'rede_neural':(score_mpl, mae_mpl),
                          'support_vector':(score_svr, mae_svr)}
        
        return dict_base_line
        
    
    
    def get_predict(self, X, modelo) -> np.array:
        
        
        if 'StandardScaler' in str(modelo[0]):
            
            # Faz as predições do número de inscritos no ENEM 2019
            previsoes = modelo[-1].predict(modelo[1].transform(X))
            previsoes = modelo[1].inverse_transform(previsoes)
        
        else:
            previsoes = modelo[-1].predict(X)       
            
        return previsoes        
        
    
    
    def get_validation(self, X, y, pre, random_state, base, scaled, plt_text, config, kwargs, validation, plot) -> np.matrix:

        """
        Validação dos modelos com StratifiedKFold(n_splits=10) com a comparação de 30 resultados de cada modelo"""
        
        resultados = {}
        random_state=None
        for modelo in self.lst_modelo:   
            
            # Cross validation Regressão com Árvore de Decisão
            resultados[modelo] = self.get_metrics(
                X, y, pre, random_state, base, scaled, plt_text, config, kwargs, validation, plot, modelo=modelo)
            
        results = np.c_[resultados['arvore'], resultados['linear_poly'], resultados['rede_neural'], resultados['support_vector']]
        
        ranks, names = self.control_validator.compare(results, self.lst_modelo)       
        
        validation = False
        names = {name.split(' - ')[0]: float(name.split(' - ')[1]) for name in names}
        result_rank = sorted(names, key=names.get)
        modelos = {}
        for i in range(2):
            
            # Seleciona o melhor modelo de acordo com as comparações feitas acima pelo método de rankeamento Friedman-Nemenyi 
            modelo = self.get_metrics(
                X, y, pre, random_state, base, scaled, plt_text, config, kwargs, validation, plot, modelo=result_rank[i], selection=True) 
            
            modelos[result_rank[i]] = modelo
            
        return results, modelos