# -*- coding: utf-8 -*-

from model.validator import Validator

from control.control_analise import ControlAnalise

class ControlValidator(object):
    
    
    def __init__(self):        
        
        self.validator          = Validator()
        self.control_analise    = ControlAnalise()
        self.resultados         = []        
        
    
    def validate(self, X, y, regressor):
        # Retorna o array com os resutltadso Score do modelo específico
        return self.validator.validate_models(X, y, regressor, self.resultados)
    
    
    def compare(self, results, lst_models):

        # self.validation = self.control_regressor.get_validation(
        #    self.X, self.y, test_size, validation, base, plt_text, balance, kwargs)                 # Invoca a função de validação StratifiedKFold
    
        # Invoca a função de comparação dos resultados de 30 testes cada modelo
        result_wilcox, result_fried, models_par, ranks, names, cds, average_ranks = self.validator.compare_results(
            results, lst_models)
        
        self.control_analise.print_conclusao(
            result_wilcox, result_fried, ranks, models_par, names, cds, average_ranks, lst_models)                           # Imprime a conclusão das comparações de validação

        return ranks, names