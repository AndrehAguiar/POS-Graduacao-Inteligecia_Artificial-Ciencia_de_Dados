# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:44:17 2020

@author: TOP Artes
"""

from sklearn.preprocessing import PolynomialFeatures,StandardScaler

class PreProcessor(object):
    
    
    def __init__(self):
        
        self.scaler         = StandardScaler()
        self.poly           = PolynomialFeatures        
        

    def scale_data(self, X, y):
        
        # Instancia a classe StandardScaler        
        scaler_X            = self.scaler
        X_train             = scaler_X.fit_transform(X[0])      # Escalona os dados previsores
        X_test             = scaler_X.transform(X[1])            # Escalona os dados previsores
        
        # Instancia a classe StandardScaler        
        scaler_y            = self.scaler 
        y_train             = scaler_y.fit_transform(y[0].ravel())      # Escalona os dados alvos
        y_test              = scaler_y.transform(y[1].ravel())          # Escalona os dados alvos  
        
        return (X_train, X_test), (y_train, y_test), scaler_X, scaler_y 
        
    
    def poly_data(self, X):        
        
        # Instancia a classe PolynomialFeatures
        poly                = self.poly(degree=1, interaction_only=True, order='F') # str in {‘C’, ‘F’}, default ‘C
        X_train_poly        = poly.fit_transform(X[0]) # Transforma os valores polinomiais
        X_test_poly         = poly.transform(X[1])
        
        return X_train_poly, X_test_poly, poly