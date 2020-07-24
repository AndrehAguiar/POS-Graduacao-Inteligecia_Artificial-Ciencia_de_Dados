# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:46:11 2020

@author: TOP Artes
"""
import pandas as pd
import numpy as np

from model.estado import Estado
from model.pre_processor import PreProcessor
from control.control_regressor import ControlRegressor

class ControlEstado:
    
    
    def __init__(self):
        
        self.estado             = Estado()      
        self.lst_targets        = list(['Inc_ENEM', 'Mediana_CN', 'Mediana_CH', 'Mediana_LN', 'Mediana_MT', 'Mediana_RD'])         
        self.pre_processor      = PreProcessor()
        self.control_regressor  = ControlRegressor()
        self.DataFrame          = pd.DataFrame()
        self.previsoes          = []
        
        
        
    def get_raw(self) -> pd.DataFrame:
        
        # Invoca a função passando a lista os anos analisados neste projeto
        DataFrame = self.estado.gerar_dados(['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'])
        
        # Invoca a função enviando a lista de colunas numéricas(Int) e o DataFrame para tratar os dados.
        self.DataFrame = self.estado.tratar_dados(list(DataFrame.columns[4:27]))
        
        return self.DataFrame
    
    
    
    def balance_data(self, DataFrame) -> pd.DataFrame:
        
        # estado = Estado()
        
        # Invoca a função passando a lista de tuplas com os indices de intervalos das colunas
        # Respectivamente Matrículas, Escolas, Docentes
        DataFrame = self.estado.calcular_totais([(6,13),(13,20),(20,27)], DataFrame.copy())
        
        # Invoca a função passando a lista de tuplas com os indices de intervalos das colunas
        # Respectivamente Área Territorial - km²[0], Pop_estimada[0][1], Escolas[0][1], Matrículas[0][1], Docentes[0][1]
        DataFrame = self.estado.calcular_densidade([(3,4),(3,33),(3,32),(3,34)], DataFrame.copy()) # Densidade por km²
        
        # Invoca a função passando a lista de tuplas com os indices de intervalos das colunas
        # Respectivamente:
        # Habitante/Inscrições ENEM[(X,Y)]
        # Habitante/Total de escolas[(X,Y)]
        # Total de Matrículas/Escolas[(X,Y)]
        # Total de Docentes/Matrículas[(X,Y)]
        DataFrame = self.estado.calcular_proporcao([(4,5),(4,33),(32,33),(32,34),(34,33)], DataFrame.copy()) # Proporção do valores absolutos
        return DataFrame
    
    

    def get_targets(self):
        return self.lst_targets
    
        
    
    def set_target(self, DataFrame) -> list:
        
        self.target = [i for i, col in enumerate(DataFrame.columns) if col == self.lst_targets[0]][0]
        
        return self.target 
    
    
    
    def get_previsores_alvos(self, DataFrame, c_target, r_target, target, balanced) -> np.array:
        
        # Define a fatia com os dados para treinamento e teste
        X = DataFrame[DataFrame[c_target]!=r_target].copy()
        
        # Define os dados alvos
        y = X.iloc[:,[target]].values
        
        cols = ['Região Geográfica','Unidade da Federação']+self.lst_targets
        if balanced:
            cols.append('POP/INSC_ENEM')
            
        # Seleciona as features relevantes para predição
        X = X.drop(cols, inplace=False, axis=1)
        
        # Define dados previsores
        X = X.iloc[:,0:].values
        
        return X, y 
    
    
    
    def get_previsores(self, DataFrame, balanced, modelo) -> np.array:
        
        # Seleciona os dados preditores de 2019
        X = DataFrame[DataFrame['ano']==2019].copy()
        
        
        # Seleciona as features relevantes para predição
        cols = ['Região Geográfica', 'Unidade da Federação']+self.lst_targets
        
        if balanced:
            cols.append('POP/INSC_ENEM')
            
        # Seleciona as features relevantes para predição
        X = X.drop(cols, inplace=False, axis=1)
        
        # Define dados previsores
        X = X.iloc[:,1:].values
                                                                                    ## Recebe as previsões  
        #self.set_previsoes(DataFrame, 'ano', 2019, self.lst_targets[0], previsoes)
        return X
    
    
    
    def set_previsoes(self, DataFrame, c_index, r_target, c_target, previsoes) -> pd.DataFrame:
        
        
        df_balanced = DataFrame.copy()
        
        df_balanced.loc[df_balanced[c_index] == r_target, c_target] = previsoes            
        #df_balanced = self.balance_data(df_balanced)                                  # Invoca a função enviando o DataFrame original
        
        return df_balanced 
    
    
    
    def pre_process(self, df, test_size, random_state, base, balanced, scaled, balance, baseline, scale, plot, validation):
        
        config = {}
        
        DataFrame = df[1].copy() if balanced else df[0].copy()
        
        plt_text = self.lst_targets[0]
        
        kwargs = self.control_regressor.set_kwargs(
            base, baseline)
        
        if len(balance + baseline + scale) > 0:    
            return self.check_options(
                df, test_size, random_state, base, balance, baseline, scale, plt_text, kwargs, validation)
    
        X, y = self.get_previsores_alvos(
            DataFrame, 'ano', 2019, self.target, balanced)                          # Separa os previsores balanceados dos alvos
            
        X_train, X_test, y_train, y_test = self.control_regressor.train_test(
            X, y, test_size, random_state)
        
        X_scaled, y_scaled, scaler_X, scaler_y = self.pre_processor.scale_data(
            (X_train[:,1:], X_test[:,1:]), (y_train, y_test))
                
        X_train_poly, X_test_poly, poly = self.pre_processor.poly_data(
            (X_train[:,1:], X_test[:,1:]))
        
        pre = {'scaler':[X_scaled, y_scaled, scaler_X, scaler_y],
               'poly':[(X_train_poly, X_test_poly), poly]}
        
        X, y, pre = (X_train[:,1:], X_test[:,1:], X_test), (y_train, y_test), pre           
        
        return self.control_regressor.get_metrics(
            X, y, pre, random_state, base, scaled, plt_text, config, kwargs, validation, plot)
    
        
    
    def check_options(self, df, test_size, random_state, base, balance, baseline, scale, plt_text, kwargs, validation):
                    
        config = {}
        
        for k in range(4):
            balanced = True if k in balance else False
            scaled = True if k in scale else False
            data_set = df[1].copy() if k in balance else df[0].copy()
            
            X, y = self.get_previsores_alvos(
                data_set, 'ano', 2019, self.target, balanced)  # Separa os previsores balanceados dos alvos
            
            pre, config[k] = self.check_base(
                X, y, k, test_size, random_state, base, balanced, scaled, plt_text, kwargs)
        
        if validation:
            return self.control_regressor.get_validation(
                X, y, pre, random_state, base, scaled, plt_text, config, kwargs, validation, plot=False)
            
        return self.control_regressor.get_metrics(
            X, y, pre, random_state, base, scaled, plt_text, config, kwargs, validation, True)
    
    
        
    def check_base(self, X, y, k, test_size, random_state, base, balanced, scaled, plt_text, kwargs):
        
        X_train, X_test, y_train, y_test = self.control_regressor.train_test(
               X, y, test_size, random_state)
                
        X_scaled, y_scaled, scaler_X, scaler_y = self.pre_processor.scale_data(
            (X_train[:,1:], X_test[:,1:]), (y_train, y_test))
        
        X_train_poly, X_test_poly, poly = self.pre_processor.poly_data(
            (X_train[:,1:], X_test[:,1:]))
        
        pre = {'scaler':[X_scaled, y_scaled, scaler_X, scaler_y],
               'poly':[(X_train_poly, X_test_poly, ), poly]}
        
        config = {'X':(X_train[:,1:], X_test[:,1:], X_test),
                     'y':(y_train.reshape(-1,1), y_test.reshape(-1,1)), 'pre':pre, 'random_state':random_state,
                     'base':base, 'scaled':scaled, 'plt_text':plt_text, 'kwargs':kwargs}
        
        return pre, config 
        
    
    
    
    