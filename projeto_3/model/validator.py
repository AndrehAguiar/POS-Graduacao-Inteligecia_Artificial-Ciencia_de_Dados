# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:37:22 2020

@author: TOP Artes
"""
import numpy as np

from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from Orange.evaluation import compute_CD, graph_ranks
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error



class Validator:    
    
    
    def __init__(self):
        
        self.kfold              = StratifiedKFold
        self.wilcoxon           = wilcoxon
        self.friedmanchisquare  = friedmanchisquare
        self.rankdata           = rankdata
        self.resultados_mean    = list([])
        
        
        
    def get_results(self, X, y, models):
        
        dct_results = {}
        
        dct_results['tree']     = self.validate_models(X,y,models[0])
        dct_results['forest']   = self.validate_models(X,y,models[1])
        dct_results['svr']      = self.validate_models(X,y,models[2])
        dct_results['mlp']      = self.validate_models(X,y,models[3])
        dct_results['gr']       = self.validate_models(X,y,models[4])
        
        results = np.c_[dct_results['tree'], dct_results['forest'], dct_results['svr'], dct_results['mlp'], dct_results['gr']]
        
        return results, dct_results
    
    
    
    def validate_models(self, X, y, model, resultados=[]) -> np.array:
        self.resultados_mean    = []
        
        for i in range(0,30):
            
            kfold = self.kfold(n_splits=10, shuffle=True, random_state=i)
            
            for indice_treinamento, indice_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
                
                model.fit(X[indice_treinamento], y[indice_treinamento].ravel())
                # print(model.score(X[indice_teste], y[indice_teste].ravel()))
                mae = mean_absolute_error(y[indice_teste], model.predict(X[indice_teste]))
        
            resultados.append(mae*(-1))
            result = np.asarray(resultados)
            self.resultados_mean.append(result.mean())
        
        return self.resultados_mean
    
    
        
    def wilcoxon_method(self, df_results):
        
        models_par = {}
        
        results = df_results.mean().sort_values(ascending=False).head(2)
        
        models_par[results.index[0]] = results[results.index[0]]
        models_par[results.index[1]] = results[results.index[1]]
        
        result_wilcox = wilcoxon(df_results[results.index[0]],df_results[results.index[1]],zero_method='pratt')
        
        return result_wilcox, models_par
    
    
    
    
    def compare_results(self, results, lst_models):
        
        # wil_result, models_par = self.wilcoxon_method(results, lst_models)
        
        fried_result = self.friedmanchisquare(*results)
        
        ranks = np.array([self.rankdata(-p) for p in results])
        
        # Calculating the average ranks.
        average_ranks = np.mean(ranks, axis=0)

        names = [lst_models[i]+' - '+str(round(average_ranks[i], 3)) for i in range(len(average_ranks))]
        
        # This method computes the critical difference for Nemenyi test with alpha=0.1.
        # For some reason, this method only accepts alpha='0.05' or alpha='0.1'.
        cd = compute_CD(average_ranks, n=len(results),alpha='0.05', test='nemenyi')
        
        # This method computes the critical difference for Bonferroni-Dunn test with alpha=0.05.
        # For some reason, this method only accepts alpha='0.05' or alpha='0.1'.
        cd1 = compute_CD(average_ranks, n=len(results), alpha='0.05', test='bonferroni-dunn')
        
        return fried_result, ranks, names, (cd, cd1), average_ranks
    
    
    
    def plot_comparisons(self, fried_result, names, cd, cd1, average_ranks):
    
        # This method generates the plot.
        graph_ranks(average_ranks, names=names,
                        cd=cd, width=10, textspace=1.5)
        
        plt.title(f'Friedman-Nemenyi={round(fried_result.pvalue, 4)}\nCD={round(cd, 3)}')
        plt.show()
        
        # This method generates the plot.
        graph_ranks(average_ranks, names=names,
                        cd=cd1, cdmethod=0, width=10, textspace=1.5)
        plt.title(f'Bonferroni-Dunn\nCD={round(cd1, 3)}')
        plt.show()
    
        return
    
    
            
    def visualizar_resultados_validacao(self, wilcox, friedman, models_par, cds, average_ranks, algorithms):
        
        print('\n'.join('{} average rank: {}'.format(a, r) for a, r in zip(algorithms, average_ranks)))
        
        par = 'não são equivalentes' if wilcox.pvalue >= 0.05 else 'são equivalentes'
        # Imprime a conclusão da comparação
        print(f"\nDe acordo com o resultado do 'Wilcoxon signed-rank' com o p-value = {round(wilcox.pvalue, 4)}.\n\
Os modelos treinados:{list(models_par.items())[0]} e {list(models_par.items())[1]} {par}.\n\
Considerando o nível de significância de (α) = 0.05.\n\n\
'The Wilcoxon signed-rank test was not designed to compare multiple random variables.\n\
So, when comparing multiple classifiers, an 'intuitive' approach would be to apply the Wilcoxon test to all possible pairs.\n\
However, when multiple tests are conducted, some of them will reject the null hypothesis only by chance (Demšar, 2006).\n\
For the comparison of multiple classifiers, Demšar (2006) recommends the Friedman test.'\n\n")
        
        rank = 'não são equivalentes' if friedman.pvalue <= 0.05 else 'são equivalentes'
        print(f"O teste de Friedman calculou o p-value = {round(friedman.pvalue, 4)}.\n\
Considerando o nível de significância de (α) = 0.05, todos os modelos {rank}.\n\
Tendo em vista o Critical Distance (CD), somente os modelos com a diferença entre as médias maior que, {cds[0]} Friedman-Nemenyi / {cds[1]} bonferroni-dunn podem ser considerados pior(es) e melhor(es).\n\n\
'Considering that the null-hypothesis was rejected, we usually have two scenarios for a post-hoc test (Demšar, 2006):\n\
All classifiers are compared to each other. In this case we apply the Nemenyi post-hoc test.\n\
All classifiers are compared to a control classifier. In this scenario we apply the Bonferroni-Dunn post-hoc test.'")
        
        print('Os testes foram realizados para validar classificadores no caso de uso citado por Demšar em 2006. Sendo os mesmos critérios de avaliação, maior melhor, a métrica foi adaptada para obedecer esses critérios e obter os resultados corretos. O "Mean Absolute Error" foi convertido para "Negative Mean Absolute Error".')
        
        return