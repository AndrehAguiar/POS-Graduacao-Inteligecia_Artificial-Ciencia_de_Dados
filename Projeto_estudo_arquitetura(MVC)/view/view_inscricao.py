# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 08:12:15 2020

@author: TOP Artes
"""
# Importa as bibliotecas de estruturação e visualização dos dados
import pandas as pd 
import numpy as np

# Importa as classes de controle de cada objeto
from control.control_estado import ControlEstado
from control.control_municipio import ControlMunicipio
from control.control_analise import ControlAnalise
from control.control_regressor import ControlRegressor


class Inscricao(object):
    
    
    def __init__(self, regiao):      
        
        
        self.regiao                 = str(regiao)
        self.control_estado         = ControlEstado()                                       # Instancia a classe de controle dos dados UF
        self.control_municipio      = ControlMunicipio()                                    # Instancia a classe de controle dos dados Municipais
        self.control_analise        = ControlAnalise()                                      # Instancia a classe de controle de análise exploratória(UF)
        
        self.control_regressor      = ControlRegressor()

        self.df                     = pd.DataFrame()
        self.df_balanced            = pd.DataFrame()
        self.df_metrics             = pd.DataFrame()
        
        self.target                 = int
        
        self.dct_baseline           = dict({})
        self.dct_baseline_sc        = dict({})
        self.dct_balanced_base      = dict({})
        self.dct_balanced_base_sc   = dict({})
        self.dct_basetunned         = dict({})
        self.dct_basetunned_sc      = dict({})
        self.dct_balanced_tunned    = dict({})
        self.dct_balanced_tunned_sc = dict({})
        self.dct_compare            = dict({})
        self.modelo                 = tuple()
                                                                                             
        if self.regiao == 'estadual':
            self.control = self.control_estado
            
        elif self.regiao == 'municipio':
            self.control = self.control_municipio
            
            
            
            
    def set_dataframe(self):
        
        self.df             = self.control.get_raw()                                     # Invoca a função de leitura dos CSVs
                                                                                         ## Recebe o DataFrame com os dados originais
        self.df_balanced    = self.control.balance_data(self.df)                         # Invoca a função enviando o DataFrame original
                                                                                         ## Recebe o DataFrame com os dados originais
        self.target         = self.control.set_target(self.df)                           # Seleciona o target
        
        return
    
    
    
    def set_predicted(self, predicted=False):
            
        self.df_balanced    = self.control.balance_data()                                # Invoca a função enviando o DataFrame original
        return    
    
    
    
    def set_metrics(self, test_size=0.15, random_state=0, base=False, balanced=False, scaled=False, balance=[], baseline=[], scale=[], plot=True, validation=False):
        
        Dataframe = [self.df, self.df_balanced]
        
        dct_metrics = self.control.pre_process(
            Dataframe, test_size, random_state, base, balanced, scaled, balance, baseline, scale, plot, validation)
        
        if validation:
            self.modelo = dct_metrics
            return self.modelo
        
        if len(balance) > 0:
            self.dct_compare                    = dct_metrics
            return
        
        if base:            
            if balanced:
                if scaled:
                    self.dct_balanced_base_sc   = dct_metrics
                    return
                self.dct_balanced_base          = dct_metrics
                return                
            else:
                if scaled:
                    self.dct_baseline_sc        = dct_metrics
                    return
                self.dct_baseline               = dct_metrics
                return        
        else:      
            if balanced:
                if scaled:
                    self.dct_balanced_tunned_sc = dct_metrics
                    return
                self.dct_balanced_tunned        = dct_metrics                                                                                     
                return
            else:
                if scaled:
                    self.dct_basetunned_sc      = dct_metrics
                    return
                self.dct_basetunned             = dct_metrics
                return                                                                                         
        return                                                                      ## Recebe o DataFrame com os dados originais
         

    
    def set_predict(self, modelo, balanced=False):        
        
        previsores = self.control.get_previsores(self.df_balanced, balanced, modelo)                                            # Invoca a função enviando o DataFrame pré-processado
           
                                                                            ## Recebe os previsores de 2019       
                                                                             
        
        
        previsoes = self.control_regressor.get_predict(previsores, modelo)              # Invoca a função enviando os previsores e o modelo selecionado
                
        return previsoes  
            
            
        self.df_balanced = self.control.set_previsoes(
            self.df_balanced, 'ano', 2019, self.control.lst_targets[0], previsoes)                           # Insere as previsões no DataFrame enviando o DataFrame, Coluna Index, Valor Query, Coluna Target e Previsões
                                                                                   ## Recebe o DataFrame com as previsões
        return 
    
    
    
    def view_metrics(self):
        
        """
        Cria a estrutura do DataFrame de métricas para comparação"""
        
        self.df_metrics['model']                    = [model for model in list(['arvore',
                                                                                  'linear_poly',
                                                                                  'rede_neural',
                                                                                  'support_vector'])]
        
        self.df_metrics['baseline(mae)']            = [model[1][1] for model in list(self.dct_baseline.items())]
        self.df_metrics['baseline_sc(mae)']         = [model[1][1] for model in list(self.dct_baseline_sc.items())]        
        self.df_metrics['balanced(mae)']            = [model[1][1] for model in list(self.dct_balanced_base.items())]
        self.df_metrics['balanced_sc(mae)']         = [model[1][1] for model in list(self.dct_balanced_base_sc.items())]        
        self.df_metrics['tunned(mae)']              = [model[1][1] for model in list(self.dct_basetunned.items())]
        self.df_metrics['tunned_sc(mae)']           = [model[1][1] for model in list(self.dct_basetunned_sc.items())]        
        self.df_metrics['balanced_tunned(mae)']     = [model[1][1] for model in list(self.dct_balanced_tunned.items())]
        self.df_metrics['balanced_tunned_sc(mae)']  = [model[1][1] for model in list(self.dct_balanced_tunned_sc.items())]
        
        self.df_metrics['compare(mae)']             = [model[1][1] for model in list(self.dct_compare.items())]
        
        self.df_metrics['baseline(score)']          = [model[1][0] for model in list(self.dct_baseline.items())]
        self.df_metrics['baseline_sc(score)']       = [model[1][0] for model in list(self.dct_baseline_sc.items())]        
        self.df_metrics['balanced(score)']          = [model[1][0] for model in list(self.dct_balanced_base.items())]
        self.df_metrics['balanced_sc(score)']       = [model[1][0] for model in list(self.dct_balanced_base_sc.items())]        
        self.df_metrics['tunned(score)']            = [model[1][0] for model in list(self.dct_basetunned.items())]
        self.df_metrics['tunned_sc(score)']         = [model[1][0] for model in list(self.dct_basetunned_sc.items())]        
        self.df_metrics['balanced_tunned(score)']   = [model[1][0] for model in list(self.dct_balanced_tunned.items())]
        self.df_metrics['balanced_tunned_sc(score)']= [model[1][0] for model in list(self.dct_balanced_tunned_sc.items())]
        
        self.df_metrics['compare(score)']           = [model[1][0] for model in list(self.dct_compare.items())]
        
        return
            
    def view_plot(self, data, predicted=False):
        
        if predicted or data == 'estrutura':
            df = self.df_balanced
        else:
            df = self.df
        
        if data == 'inscricao':
            self.control_analise.inscritos_ano(df, predicted)                       # Plota distribuição de Inscrições no ENEM por ANO(2010 a 2019)/UF
            return
        
        if data == 'estrutura':
            self.control_analise.estrutura_ano(df, predicted)                       # Plota distribuição da estrutura educacional ANO(2010 a 2019)/UF
            return
        
        print("Por getileza, especifique os dados que deseja plotar ('inscricao' ou 'estrutura')")
        
        return
        