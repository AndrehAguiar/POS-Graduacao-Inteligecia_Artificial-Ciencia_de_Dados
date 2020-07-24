# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:16:14 2020

@author: TOP Artes
"""

import pandas as pd
import numpy as np

from model.municipio import Municipio

class ControlMunicipio(object):
    
    
    def __init__(self):
        
        self.municipio      = Municipio()      
        self.lst_targets    = list(['Inc_ENEM', 'Mediana_CN', 'Mediana_CH', 'Mediana_LN', 'Mediana_MT', 'Mediana_RD'])
    

    def get_targets(self):
        return self.lst_targets
        
    
    def set_target(self, DataFrame) -> list:        
        return [i for i, col in enumerate(DataFrame.columns) if col == self.lst_targets[0]][0]    
    
        
    def get_raw(self) -> pd.DataFrame:
        
        # Invoca a função passando a lista os anos analisados neste projeto
        self.desempenho_mun = self.municipio.gerar_dados(['2014','2015','2016','2017','2018','2019'])
        
        # Invoca a função enviando a lista de colunas numéricas(Int) e o DataFrame para tratar os dados.
        self.desempenho_mun = self.municipio.tratar_dados(list(self.desempenho_mun.columns[4:]))
        
        return self.desempenho_mun
    
    
    def get_previsores_alvos(self, DataFrame, c_target, r_target, balanced, target) -> np.array:
        
        self.desempenho_mun = DataFrame
        
        # Define a fatia com os dados para treinamento e teste
        X = self.desempenho_mun.copy()
        
        # Define os dados alvos
        y = X.iloc[:,[target]].values
        
        cols = ['ano', 'COD. MUNIC', 'NOME_MUNICÍPIO']+self.lst_targets
            
        # Seleciona as features relevantes para predição
        X = X.drop(cols, inplace=False, axis=1)
        
        # Define dados previsores
        X = X.iloc[:,0:].values
        
        return X, y    
    
    
    def get_previsores(self, DataFrame, balanced=False) -> np.array:
        
        self.desempenho_mun = DataFrame
        
        # Seleciona os dados preditores de 2019
        X = self.desempenho_mun.copy()
        
        # Seleciona as features relevantes para predição
        cols = ['ano', 'COD. MUNIC', 'NOME_MUNICÍPIO']
            
        # Seleciona as features relevantes para predição
        X = X.drop(cols, inplace=False, axis=1)
        
        # Define dados previsores
        X = X.iloc[:,0:].values
        
        return X
    
    def process_data(self, DataFrame) -> pd.DataFrame:
        
        # estado = Estado()
        
        self.desempenho_mun = DataFrame.copy()
        
        # Invoca a função passando a lista de tuplas com os indices de intervalos das colunas
        # Respectivamente Matrículas, Escolas, Docentes
        self.desempenho_mun = self.municipio.calcular_totais([(5,12),(12,19),(19,-1)], self.desempenho_mun)
        
        # Invoca a função passando a lista de tuplas com os indices de intervalos das colunas
        # Respectivamente Área Territorial - km²[0], Pop_estimada[0][1], Escolas[0][1], Matrículas[0][1], Docentes[0][1]
        self.desempenho_mun = self.municipio.calcular_densidade([(3,4),(3,27),(3,26),(3,28)], self.desempenho_mun) # Densidade por km²
        
        # Invoca a função passando a lista de tuplas com os indices de intervalos das colunas
        # Respectivamente:
        # Habitante/Inscrições ENEM[(X,Y)]
        # Habitante/Total de escolas[(X,Y)]
        # Total de Matrículas/Escolas[(X,Y)]
        # Total de Docentes/Matrículas[(X,Y)]
        self.desempenho_mun = self.municipio.calcular_proporcao([(4,5),(4,33),(32,33),(32,34),(34,33)], self.desempenho_mun) # Proporção do valores absolutos

        return self.desempenho_mun