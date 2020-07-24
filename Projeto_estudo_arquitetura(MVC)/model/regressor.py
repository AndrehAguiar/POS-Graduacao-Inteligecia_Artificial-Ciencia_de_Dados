# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 20:55:35 2020

@author: TOP Artes
"""

# Importa a biblioteca para estruturação dos dados
import numpy as np

# Importa as bibliotecas para análise e visualização do dados

# Importa as bibliotecas dos modelos implementados
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Importa a biblioteca de métricas para comparação
from sklearn.metrics import mean_absolute_error

from control.control_analise import ControlAnalise
    
# Importa a classe de validação dos modelos
from control.control_validator import ControlValidator


class Regressor(object):    
    
    def __init__(self, X=np.array([]), y=np.array([]), ploter=list([])):
        
        self.X = X
        self.y = y
        self.ploter = ploter
        self.control_validator = ControlValidator()
        self.control_analise = ControlAnalise()
        
    
    
    def treinar_arvore(self, X, y, pre, scaled, plot, dct_text, validation, kwargs):
        
        """
        Recebe os valores previsores, targets e definição
        Retorna o vetor com as predições de valoress para tabela de 2019 """
        
        if 'boost' not in kwargs:
            kwargs['boost']={}
            
        # Instancia a classe para Regressão com Árvore de Decisão
        regressor = AdaBoostRegressor(DecisionTreeRegressor(**kwargs['tree']), **kwargs['boost'])
        
        if len(kwargs['boost']) == 0:
            kwargs = {} 
        
        if validation:
            return self.control_validator.validate(X, y, regressor)
        
        score, mae, previsoes= self.run_regressor(
            X, (y[0].ravel(), y[1].ravel()), regressor, pre, scaled)
        
        # Especifica as features mais relevantes para o modelo
        feat_importances = regressor.feature_importances_
        
        if plot:       
            if scaled:
                y = (y[0], pre['scaler'][3].inverse_transform(y[1])) 
    
                # Calcula a média absoluta dos erros
                # mae = mean_absolute_error(y[1], previsoes)            
        
            dict_modelo = {'Nome':'Regressão com Árvore de Decisão',
                           'Score':score, 'MAE':mae, 'Scale':scaled, 'kwargs':kwargs}
            self.ploter.append([X[2], y[1], previsoes, dct_text, dict_modelo])
            
            if len(self.ploter) == 4:
                self.ploter = self.control_analise.plotar_previsoes(self.ploter)
        
        return feat_importances, score, mae, regressor
    


    def treinar_linear_poly(self, X, y, pre, scaled, plot, dct_text, validation, kwargs):
        """
        Recebe os valores previsores, targets e hiperparametros
        Retorna o vetor com as predições de valoress para tabela de 2019 """
        
        # Instancia o modelo regressor
        regressor = LinearRegression(**kwargs)
        
        if validation:
            return self.control_validator.validate(X, y, regressor)            
        
        score, mae, previsoes = self.run_regressor(
            X, (y[0].ravel(), y[1].ravel()), regressor, pre, scaled)
        
        if plot: 
            if scaled:
                y = (y[0], pre['scaler'][3].inverse_transform(y[1])) 
    
                # Calcula a média absoluta dos erros
                # mae = mean_absolute_error(y[1], previsoes)                  
        
            dict_modelo = {'Nome':'Regressão Linear(Polinomial)',
                           'Score':score, 'MAE':mae, 'Scale':scaled, 'kwargs':kwargs}            
            self.ploter.append([X[2], y[1], previsoes.reshape(1,-1)[0], dct_text, dict_modelo])
            
            if len(self.ploter) == 4:
                self.ploter = self.control_analise.plotar_previsoes(self.ploter)
            
        return score, mae, regressor
    
    
    def treinar_redeneural(self, X, y, pre, scaled, plot, dct_text, validation, kwargs):
        """
        Recebe os valores previsores, targets e definição
        Retorna o vetor com as predições de valoress para tabela de 2019 """
        
        # Instancia o modelo regressor        
        regressor = MLPRegressor(**kwargs)
        
        if validation:
            return self.control_validator.validate(X, y, regressor)
        
        score, mae, previsoes = self.run_regressor(
            X, (y[0].ravel(), y[1].ravel()), regressor, pre, scaled)
        
        # Reverte os valores para escala original
        # y_test = scaler_y.inverse_transform(y_test)
        # previsoes = scaler_y.inverse_transform(previsoes)
        
        if plot:   
            # previsoes = pre['scaler'][3].inverse_transform(previsoes)
            y = (y[0], pre['scaler'][3].inverse_transform(y[1])) 
        
            if not scaled:
                previsoes = pre['scaler'][3].inverse_transform(previsoes)

                # Calcula a média absoluta dos erros
                mae = mean_absolute_error(y[1], previsoes)           
        
            dict_modelo = {'Nome':'Regressão com Rede Neural(MLP)','Score':score, 'MAE':mae, 'Scale':scaled, 'kwargs':kwargs} 
            
            self.ploter.append([X[2], y[1], previsoes, dct_text, dict_modelo])
            
            if len(self.ploter) == 4:
                self.ploter = self.control_analise.plotar_previsoes(self.ploter)
            
        return score, mae, regressor  


    def treinar_svr(self, X, y, pre, scaled, plot, dct_text, validation, kwargs):
        """
        Recebe os valores previsores, targets e definição
        Retorna o vetor com as predições de valoress para tabela de 2019 """
        
        # Instancia o modelo regressor        
        regressor = SVR(**kwargs) 
        
        if validation:
            return self.control_validator.validate(X, y, regressor)
        
        score, mae, previsoes = self.run_regressor(
            X, (y[0].ravel(), y[1].ravel()), regressor, pre, scaled)
        
        # Reverte os valores para escala original
        # y_test = scaler_y.inverse_transform(y_test)
        # previsoes = scaler_y.inverse_transform(previsoes)
        
        if plot:
            
            y = (y[0], pre['scaler'][3].inverse_transform(y[1]))
            
            if not scaled:
                previsoes = pre['scaler'][3].inverse_transform(previsoes)
            #previsoes = previsoes.reshape(1, -1)
        
                # Calcula a média absoluta dos erros
                mae = mean_absolute_error(y[1], previsoes)         
        
            dict_modelo = {'Nome':'Support Vector Regression', 'Score':score, 'MAE':mae, 'Scale':scaled, 'kwargs':kwargs}       
            self.ploter.append([X[2], y[1].reshape(-1,1), previsoes, dct_text, dict_modelo]) 
            
            if len(self.ploter) == 4:
                self.ploter = self.control_analise.plotar_previsoes(self.ploter)
            
        return score, mae, regressor  
    
    
    def run_regressor(self, X, y, regressor, pre, scaled):
        
        """
        Recebe os valores previsores, targets e definição
        Retorna o vetor com as predições de valoress para tabela de 2019 """ 
            
        # if scaled:
        #    X = pre['scaler'][0]
        #    y = pre['scaler'][1]         
        
        # Treina o modelo
        
        regressor.fit(X[0], y[0])
            
        # Calcula o índice de acertos do modelo
        score = regressor.score(X[1], y[1])
        
        # Faz as predições dos targets especificados
        previsoes = regressor.predict(X[1])
        
        # Calcula a média absoluta dos erros
        mae = mean_absolute_error(y[1], previsoes)
        
        if scaled:
            previsoes = pre['scaler'][3].inverse_transform(previsoes)
            y = pre['scaler'][3].inverse_transform(y[1])
        
            # Calcula a média absoluta dos erros
            mae = mean_absolute_error(y, previsoes)
        
        return score, mae, previsoes
