# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:55:54 2020

@author: TOP Artes
"""
# Importa a classe Inscrição (Dados de Inscritos no ENEM/UF)
from view.view_inscricao import Inscricao
inscricao_uf = Inscricao('estadual')                                   # Istancia a Classe Inscrição
inscricao_uf.set_dataframe()                                           # Carrega o Dataframe com os dados originais                            

df1 = inscricao_uf.df
inscricao_uf.df_balanced.info()
inscricao_uf.X
inscricao_uf.set_metrics(base=False, balanced=True, scaled=True)       # BaseLine


inscricao_uf.view_metrics()                                            # Cria o DataFrame com as métricas tiradas dos modelos    

inscricao_uf.df_metrics[inscricao_uf.df_metrics.columns[[range(9)]]]                       # Imprime os Scores

inscricao_uf.df_metrics[inscricao_uf.df_metrics.columns[[range(9,18)]]]                    # Imprime as MAEs

"""
baseline: Hiperparâmetros padrões
balance: DataFrame com os dados balanceados
scaled: StandardScaler
0 = Arvore
1 = Linear Polinomial
2 = Rede Neural
3 = Support Vetor"""
inscricao_uf.set_metrics(baseline = [1], scale = [0,2,3], balance = [0,1,2,3])            # Plota cada modelo com as configurações especificas
                   

results, modelo = inscricao_uf.set_metrics(validation=True, baseline = [1], scale = [0,2,3], balance = [0,1,2,3])            # Faz a validação de acordo com os 30 testes
                                                                       ## StratifiedKFold - Friedman-Nemenyi/Bonferroni-Bunn
inscricao_uf.previsoes                                                                 
results.mean(axis=0)   

str(modelo['linear_poly'][0]) == 'StandardScaler()'

modelo['rede_neural'][1].inverse_transform(modelo['rede_neural'][-1].predict(modelo['rede_neural'][0].transform(X.ravel())))

previsoes = inscricao_uf.set_predict(modelo['rede_neural'], balanced=True)

inscricao_uf.view_plot('inscricao', predicted=False)                     # Plota distribuição de Inscrições no ENEM por ANO(2010 a 2019)/UF

### Necessário o balanceamento
inscricao_uf.view_plot('estrutura', predicted=False)   
                           
inscricao_uf.df_balanced.query('ano==2019')['Inc_ENEM']                 # Imprime a quantidade de inscrições previstas para cada estado
inscricao_uf.df_balanced.info()
