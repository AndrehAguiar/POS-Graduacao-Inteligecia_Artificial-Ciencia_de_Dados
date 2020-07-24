# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:47:40 2020

@author: TOP Artes
"""

from model.analise import Analise


class ControlAnalise:
    
    
    def __init__(self):
        
        self.analise = Analise()
        
    def set_plt_text(self, plt_text):
        
        dct_text = {}
        
        if plt_text == 'Inc_ENEM':
        
            dct_text['ylim']            = '(min(y_test)-4500),(max(y_test)+300000)'
            dct_text['yscale']          = 'log'
            dct_text['ylabel']          = 'Quantidade de Inscrições'
            dct_text['title']           = 'da quantidade de inscrições anuais'
            
        else:
        
            dct_text['ylim']            = '(min(y_test)-10),(max(y_test)+10)'
            dct_text['yscale']          = 'linear'
            dct_text['ylabel']          = 'Mediana estaduasl das notas'
        
            if plt_text == 'Mediana_CN':
                dct_text['title']       = 'das notas de Ciências da Naturaza'
        
            if plt_text == 'Mediana_CH':
                dct_text['title']       = 'das notas de Ciências Humanas'
        
            if plt_text == 'Mediana_LN':
                dct_text['title']       = 'das notas de Linguagens'
        
            if plt_text == 'Mediana_MT':
                dct_text['title']       = 'das notas de Matemática'
        
            if plt_text == 'Mediana_RD':
                dct_text['title']       = 'das notas de Redação'
        
        return dct_text
    
    
    def inscritos_ano(self, DataFrame, predicted):
        
        if predicted:
            df = DataFrame
            regressor = str(modelo).split('(')[0]
            dct_plot = {'title':f'Gráfico de dispersão da quantidade de inscrições anuais por estado.\n\
                Dados originais do ENEM de 2010 a 2018 e 2019 estimado por {regressor}'}
            
        else:        
            df = DataFrame[DataFrame['ano']!=2019]
            dct_plot = {'title':'Gráfico de dispersão da quantidade de inscrições anuais por estado.\n\
                Dados originais do ENEM de 2010 a 2018'}
    
        # Invoca a função para plotagem do gráfico temporal de inscrições no ENEM(2010 a 2018) estadual
        self.analise.visualizar_inscritos_ano(df, dct_plot)
        
        return
    
    def estrutura_ano(self, DataFrame, predicted): 
        
        if predicted:
            regressor = str(modelo).split('(')[0]       
            lst_anos = [2017, 2018, 2019]
            df = DataFrame[DataFrame['ano']>=2017]
            dct_plot = {'regressor':regressor}
            
        else:
            lst_anos = list(DataFrame[DataFrame['ano']!=2019]['ano'].unique())
            df = DataFrame
            dct_plot = {'title':'Histograma da estrutura educacional de cada estado.\n\
                Dados originais do ENEM de 2010 a 2018'}
    
        
        # Invoca a função para plotagem dos gráficos de estrutura educacional por ano(2010 a 2018) estadual
        self.analise.visualizar_proporcoes(df, lst_anos, dct_plot)
        
        return
        
    
    def plotar_previsoes(self, list):

        """
        Recebe os dados previsores e alvos originais, os de teste e a previsão(X_test) do modelo.
        Imprime o gráfico."""
            
        ploter = list
        self.analise.visualizar_comparacao(ploter)
        ploter = [] 
         
        return ploter
    
    def print_conclusao(self, wilcox, friedman, ranks, models_par, names, cds, average_ranks, lst_models):
        
        self.analise.plot_comparisons(friedman, names, cds[0], cds[1], average_ranks)        
        self.analise.visualizar_resultados_validacao(wilcox, friedman, models_par, cds, average_ranks, lst_models)
        
        return