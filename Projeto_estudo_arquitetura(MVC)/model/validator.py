# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:37:22 2020

@author: TOP Artes
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from Orange.evaluation import compute_CD



class Validator:    
    
    
    def __init__(self, X = np.array([]), y = np.array([])):
        
        self.kfold              = StratifiedKFold
        self.wilcoxon           = wilcoxon
        self.friedmanchisquare  = friedmanchisquare
        self.rankdata           = rankdata
        self.resultados_mean    = list([])
        
        
    
    def validate_models(self, X, y, regressor, resultados) -> np.array:
        self.resultados_mean    = []
        # TODO especificar a quantidade de testes 30
        for i in range(0,3):
            
            kfold = self.kfold(n_splits=10, shuffle=True, random_state=i)
            
            for indice_treinamento, indice_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
                
                regressor.fit(X[indice_treinamento], y[indice_treinamento].ravel())
                score = regressor.score(X[indice_teste], y[indice_teste].ravel())    
        
            resultados.append(score)
            result = np.asarray(resultados)
            self.resultados_mean.append(result.mean())
        
        return self.resultados_mean
    
    
        
    def wilcoxon_method(self, results, lst_models):
        
        models_par = {}
        df_results = pd.DataFrame(columns=lst_models, data=results)
        
        results = df_results.mean().sort_values(ascending=False).head(2)
        
        models_par[results.index[0]] = results[results.index[0]]
        models_par[results.index[1]] = results[results.index[1]]
        
        """mean_models = sorted(results[:,0:].mean(axis=0), reverse=True)
        lst_mean = results[:,0:].mean(axis=0)
        par = []
        
        while len(par) < 2:
            
            par.append(np.where(lst_mean == np.max(lst_mean))[0][0])
            
            if len(par) <= 1:
                lst_mean[np.where(lst_mean == np.max(lst_mean))]=0"""
        
        wil_result = wilcoxon(df_results[results.index[0]],df_results[results.index[1]],zero_method='pratt')
        
        return wil_result, models_par
    
    
    
    def compare_results(self, results, lst_models):
        
        wil_result, models_par = self.wilcoxon_method(results, lst_models)
        
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
        
        return wil_result, fried_result, models_par, ranks, names, (cd, cd1), average_ranks
    
    
    