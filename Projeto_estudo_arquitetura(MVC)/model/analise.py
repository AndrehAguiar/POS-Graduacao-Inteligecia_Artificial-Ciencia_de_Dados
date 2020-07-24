# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:58:39 2020

@author: TOP Artes
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Orange.evaluation import graph_ranks

sns.set_style('whitegrid')
# %matplotlib inline


class Analise(object):
    
    """
    Classe responsável pelas impressões dos gráficos de análises exploratórias dos dados Estaduais e DF"""
    
    def __init__(self, X=np.array([]), y=np.array([])):
        self.X = X
        self.y = y
        
    def visualizar_inscritos_ano(self, DataFrame, dict):
        
        """
        Recebe os dados previsores, alvos e DataFrame originais.
        Imprime o Gráfico de dispersão da quantidade de inscrições anuais por estado."""
        
        dct_plot = dict
        df = DataFrame        
        # Formata os dados para geração do gráfico
        # data = np.c_[X, DataFrame.iloc[:X.shape[0],2].values, y[:,0]]
        
        # Define as dimensões da imagem
        plt.figure(figsize = (12,10))
        
        # Define os dados dos eixos e agrupamentos
        ax = sns.scatterplot(df['ano'], df['Inc_ENEM'], hue=df['Unidade da Federação'])
        sns.lineplot(
            x='ano', y='Inc_ENEM', data=DataFrame,
            hue='Unidade da Federação', legend=False)
        
        
        # Define o local e texto da fonte dos dados
        ax.text(x=-0.1, y=-0.15, s='Fonte: INSTITUTO NACIONAL DE ESTUDOS E PESQUISAS EDUCACIONAIS ANÍSIO TEIXEIRA.\n\
Sinopse Estatística da Educação Básica de 2010 a 2019. Brasília: Inep, 2020.\n\
Disponível em: <http://portal.inep.gov.br/sinopses-estatisticas-da-educacao-basica>. Acesso em: 05/06/2020',
    fontsize=9, ha='left', va='bottom', transform=ax.transAxes)
        
        # Posiciona a legenda fora do gráfico
        ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        
        # Define a escala do eixo y, os rótulos e o título do gráfico
        ax.set(yscale='log', xlabel='Ano', ylabel='Qtd. Inscrições no ENEM',
               title=dct_plot['title'])
        
        # Imprime o gráfico
        plt.show()
        
        return 
    
    def visualizar_proporcoes(self, DataFrame, list, dct_plot):
        
        """
        Recebe os dados previsores, alvos e DataFrame originais.
        Imprime o gráfico."""
        
        lst_anos = list
        ys = 60 if len(lst_anos) > 3 else 18
        plt.figure(figsize=(20,ys))    
        p = 1
        
        for ano in lst_anos:
            
            data = DataFrame[['ano','Unidade da Federação',
                            'POP/INSC_ENEM','MATRICULA/DOCENTE',
                            'DOCENTE/ESCOLA','POP/ESCOLA',
                            'MATRICULA/ESCOLA']].query(f'ano=={ano}')    
            
            ax = plt.subplot(len(lst_anos), 2, p)   
            ax.bar(data['Unidade da Federação'], data['POP/INSC_ENEM'], label='Habitantes/Insc. ENEM')
            ax.bar(data['Unidade da Federação'], data['MATRICULA/DOCENTE'], label='Matrícula/Docente')
            ax.bar(data['Unidade da Federação'], data['DOCENTE/ESCOLA'], label='Docente/Escola')
            
            ax.set_title(f'Distribuição proporcional das variáveis {ano}\n\
Habitantes / Matrículas / Docente / Escola / Insc. ENEM{({dct_plot["regressor"]}) if ano == 2019 else ""}') 
            ax.set_ylabel(f'Quantidade no ano {ano} (0 a 100)')
            ax.set_ylim(0,100)
            ax.legend()    
            
            ax.set_xticklabels(data['Unidade da Federação'], rotation=75)
            
            ax1 = plt.subplot(len(lst_anos), 2, p+1)
            ax1.bar(data['Unidade da Federação'], data['POP/ESCOLA'], label='Habitantes/Escolas')
            ax1.bar(data['Unidade da Federação'], data['MATRICULA/ESCOLA'], label='Matrículas/Escolas') 
            
            ax1.set_title(f'Distribuição proporcional das variáveis {ano}\n\
Habitantes / Matrículas / Escola') 
            ax1.set_ylabel(f'Quantidade no ano {ano} (0 a 1000)')
            ax1.legend()
            
            ax1.set_xticklabels(data['Unidade da Federação'], rotation=75)
            
            p+=2        
        
        # Define o local e texto da fonte dos dados
        ax.text(x=-0.03, y=-0.85, s='Fonte: INSTITUTO NACIONAL DE ESTUDOS E PESQUISAS EDUCACIONAIS ANÍSIO TEIXEIRA.\n'
'Sinopse Estatística da Educação Básica de 2010 a 2019. Brasília: Inep, 2020.\n'
'Disponível em: <http://portal.inep.gov.br/sinopses-estatisticas-da-educacao-basica>. Acesso em: 05/06/2020\n'

'Fonte: IBGE. Diretoria de Pesquisas - DPE -  Coordenação de População e Indicadores Sociais - COPIS.\n'
'Disponível em: <http://www.dados.gov.br/dataset/cd-censo-demografico>. Acesso em: 05/06/2020\n'
    
'Fonte: IBGE. Diretoria de Pesquisas - DPE -  Coordenação de População e Indicadores Sociais - COPIS.\n'
'Disponível em: <https://www.ibge.gov.br/geociencias/downloads-geociencias.html>. Acesso em: 05/06/2020\n',
    fontsize=9, ha='left', va='bottom', transform=ax.transAxes)
            
        plt.subplots_adjust(hspace=.6)
            
        plt.show()
        
        return 
        
    def visualizar_comparacao(self, list):
        p = 1
        
        ploter = list
        # Define as dimensões da imagem
        plt.figure(figsize = (20,10))
        
        for i, modelo in enumerate(ploter):
            
            X_test          = modelo[0]
            y_test          = modelo[1]
            previsoes       = modelo[2]
            dict_text       = modelo[3]
            dict_modelo     = modelo[4]
            
            data = 'Base Original' if X_test.shape[1] <= 24 else f'Features Processadas'
            config = 'Baseline' if len(dict_modelo['kwargs'])<=2 else 'Tunned'
            scale = ' / Escalonado' if dict_modelo['Scale'] else ''
            
            ax = plt.subplot(2, 2, p)
            # Define os dados dos eixos e agrupamentos
            sns.scatterplot(X_test[:,0], y_test[:,0],s=200, label='Alvos')
            sns.scatterplot(X_test[:,0], previsoes, marker='X',s=120, label='Predições')
            
            # Define o local e texto da fonte dos dados
            ax.text(x=0, y=-0.3, s='Fonte: INSTITUTO NACIONAL DE ESTUDOS E PESQUISAS EDUCACIONAIS ANÍSIO TEIXEIRA.\n\
Sinopse Estatística da Educação Básica de 2010 a 2019. Brasília: Inep, 2020.\n\
Disponível em: <http://portal.inep.gov.br/sinopses-estatisticas-da-educacao-basica>. Acesso em: 05/06/2020',
fontsize=7, ha='left', va='bottom', transform = ax.transAxes)
                
            ax.text(x=1., y=-0.3, s=f'Score do modelo: {round(dict_modelo["Score"], 4)}\n\
Média de Erro Absoluto {round(dict_modelo["MAE"], 4)}\n\
{dict_modelo["Nome"]}.',
fontsize=9, ha='right', va='bottom', transform = ax.transAxes)
            
            # Define a escala do eixo y, os rótulos e o título do gráfico
            ax.set(yscale           = dict_text['yscale'],
                   ylim             = eval(dict_text['ylim']),
                   ylabel           = dict_text['ylabel'],
                   xlabel           = 'Ano',
                   title            =f'{i} - Gráfico de dispersão {dict_text["title"]}\n\
Dados originais do ENEM de 2010 a 2018 X Previsões {dict_modelo["Nome"]}\n\
{data} / {config}{scale}')
                
            ax.grid(True)
            
            if p%2 == 0:
                ax.set_ylabel('')
                # Oculta os valores do eixo Y
                plt.setp(ax.get_yticklabels(), visible=False)
        
                # Posiciona a legenda fora do gráfico                 
                # ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
                
            p+=1
        plt.subplots_adjust(hspace=.6, wspace=.05)
        # Imprime o gráfico
        plt.show()
        
        return
        
    def visualizar_resultados_validacao(self, wilcox, friedman, models_par, cds, average_ranks, algorithms):
        
        print('\n'.join('{} average rank: {}'.format(a, r) for a, r in zip(algorithms, average_ranks)))
        
        par = 'não são equivalentes' if wilcox.pvalue <= 0.05 else 'são equivalentes'
        # Imprime a conclusão da comparação
        print(f"De acordo com o resultado do 'Wilcoxon signed-rank' com o p-value = {round(wilcox.pvalue, 4)}.\n\
Os modelos treinados:{list(models_par.items())[0]} e {list(models_par.items())[1]} {par}.\n\
Considerando o nível de significância de (α) = 0.05.\n\n\
'The Wilcoxon signed-rank test was not designed to compare multiple random variables.\n\
So, when comparing multiple classifiers, an 'intuitive' approach would be to apply the Wilcoxon test to all possible pairs.\n\
However, when multiple tests are conducted, some of them will reject the null hypothesis only by chance (Demšar, 2006).\n\
For the comparison of multiple classifiers, Demšar (2006) recommends the Friedman test.'\n\n")
        
        rank = 'não são equivalentes' if friedman.pvalue <= 0.05 else 'são equivalentes'
        print(f"O teste de Friedman calculou o p-value = {round(friedman.pvalue, 4)}.\n\
Considerando o nível de significância de (α) = 0.05, os modelos {rank}.\n\
Tendo em vista o Critical Distance (CD), somente os modelos com a diferença entre as médias maior que, {cds[0]} Friedman-Nemenyi / {cds[1]} bonferroni-dunn podem ser considerados pior(es) e melhor(es).\n\n\
'Considering that the null-hypothesis was rejected, we usually have two scenarios for a post-hoc test (Demšar, 2006):\n\
All classifiers are compared to each other. In this case we apply the Nemenyi post-hoc test.\n\
All classifiers are compared to a control classifier. In this scenario we apply the Bonferroni-Dunn post-hoc test.'")
        
        return
    
    
    
    def plot_comparisons(self, fried_result, names, cd, cd1, average_ranks):
        
        # This method generates the plot.
        graph_ranks(average_ranks, names=names,
                        cd=cd, width=8, textspace=1.5)
        
        plt.title(f'Friedman-Nemenyi={round(fried_result.pvalue, 4)}\nCD={round(cd, 3)}')
        plt.show()
        
        # This method generates the plot.
        graph_ranks(average_ranks, names=names,
                        cd=cd1, cdmethod=0, width=8, textspace=1.5)
        plt.title(f'Bonferroni-Dunn\nCD={round(cd1, 3)}')
        plt.show()

        return
    
    
    