# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:57:38 2020

@author: TOP Artes
"""
# Importa a biblioteca para estruturação dos dados e visualização dos dados   
import pandas as pd

class Municipio(object):
    
    def __init__(self):
        self.df = pd.DataFrame()        
            
     
    def gerar_dados(self, list) -> pd.DataFrame:
        
        """
        Recebe a lista com os anos a serem concatenados no dataframe
        Retorna o DataFrame de desempenho(MEDIANA) no ENEM de cada estado """
        
        lst_anos = list
        
        # Carrega dos dados das tabelas de 2014 a 2019        
        for ano in lst_anos:
            
            df_tmp = pd.DataFrame()
            df_tmp['ano'] = ano
            df_territorio_ano = pd.read_csv(f'CSVs/{ano}_territorio_mun.csv', sep=';', decimal=',')            
            df_territorio_ano = df_tmp.join(df_territorio_ano, how='outer')
            df_territorio_ano.columns = ['ano', 'ID', 'CD_GCUF', 'NM_UF',
                                         'NM_UF_SIGLA', 'COD. MUNIC',
                                         'NOME_MUNICÍPIO', 'AREA_km²']
            
            df_territorio_ano['ano'].fillna(int(ano), inplace=True)
            
            
            df_pop_ano = pd.read_csv(f'CSVs/{ano}_pop_mun.csv', sep=';', decimal=',')
            
            # Remove a coluna duplicada
            # Pradroniza o nome da coluna COD. MUNIC para mesclar os DataFrames
            df_pop_ano.drop(['UF','COD. UF','COD_MUNIC',
                             'NOME DO MUNICÍPIO'], axis=1, inplace=True)
            
            df_territorio_ano = pd.merge(df_territorio_ano[['ano','COD. MUNIC', 'NOME_MUNICÍPIO',
                                                            'AREA_km²']], df_pop_ano, on='COD. MUNIC')           
            
            df_educacao_ano = pd.read_csv(f'CSVs/{ano}_educacao_basica.csv', sep=';', decimal=',')            
            df_tmp = pd.merge(df_territorio_ano, df_educacao_ano, on='COD. MUNIC')
            
            self.df = pd.concat([self.df, df_tmp], ignore_index=True)
        
        return self.df
 
    
    def tratar_dados(self, list) -> pd.DataFrame:
        
        """
        Recebe a lista com as colunas e o DataaFrame de dados a serem tratados e convertidos
        Retorna o DataFrame com os dados tratados -> NaN = 0 / StringObj = Int """ 
        
        lst_colunas = list
        
        # Considera que se não há número informado não existe no estado
        # Trata valores anômalos das tabelas considerando que "NaN" é ZERO(Escolas, Matrículas, Docentes etc)        
        for col in lst_colunas:           

            # Remove indicação das anotações sobre o CENSO realizado no respectivo ano para padronização dos dados
            # Padroniza os tipos de dados da coluna POPULAÇÃO ESTIMADA convertendo para números inteiros
            self.df[col] = self.df[col].str.replace(r"\(.*?\)","")
            self.df[col] = self.df[col].str.replace(" -   ",'0')
            self.df[col] = self.df[col].str.replace(".","")
            self.df[col].fillna(0, inplace=True)
            self.df[col] = self.df[col].astype(str).astype(int)
        
        return self.df
        
    def calcular_densidade(self, list, DataFrame) -> pd.DataFrame:
        
        """
        Recebe um lista de tuplas com os indices de intervalos das colunas e o DataFrame com a estrutura de dados com as colunas na ordem padronizada
        Retorna DataFrame com a soma de escolas, matrículas e corpo docente de cada estado / ano """
        
        lst_tuples = list
        self.df = DataFrame
        
        # Calcula a densidade de população, escolas, matrículas e docentes por km² por km² de cada estado 
        # Estimativa de habitante/km²
        self.df['HAB/km²']                   = (self.df.iloc[:,lst_tuples[0][1]].transpose().values/ \
                                                           self.df.iloc[:,lst_tuples[0][0]].transpose().values)
        
        # Estimativa de Escolas/km²
        self.df['ESCOLA/km²']                = (self.df.iloc[:,lst_tuples[1][1]].values/ \
                                                           self.df.iloc[:,lst_tuples[1][0]].values)
        
        # Estimativa de habitantes(km²)/escolas(km²)
        self.df['HAB_ESCOLA/km²']            = (self.df.iloc[:,lst_tuples[1][1]].values/ \
                                                           (self.df.iloc[:,self.df.shape[1]-2].values/ \
                                                            self.df.iloc[:,self.df.shape[1]-1].values))
         
        # Estimativa de matrículas(km²)/escolas(km²)
        self.df['MATRIC_ESC/km²']            = ((self.df.iloc[:,lst_tuples[2][1]].values/ \
                                                            self.df.iloc[:,lst_tuples[2][0]].values)/ \
                                                           self.df.iloc[:,self.df.shape[1]-3].values)
         
        # Estimativa do corpo docente(km²)/escolas(km²)
        self.df['DOCENTE_ESC/km²']           = ((self.df.iloc[:,lst_tuples[3][1]].values/ \
                                                            self.df.iloc[:,lst_tuples[3][0]])/ \
                                                           self.df.iloc[:,self.df.shape[1]-4].values)
        
        # Estimativa do corpo docente(km²)/matrículas(km²)
        self.df['DOCENTE_MATRIC/km²']        = ((self.df.iloc[:,lst_tuples[3][1]].values/ \
                                                            self.df.iloc[:,lst_tuples[3][0]])/ \
                                                           (self.df.iloc[:,lst_tuples[2][1]].values/ \
                                                            self.df.iloc[:,lst_tuples[2][0]].values))
        
        return self.df
    
    
    
    def calcular_totais(self, list, DataFrame) -> pd.DataFrame:
            
        """
        Recebe um lista de tuplas com os indices de intervalos das colunas e o DataFrame com a estrutura de dados com as colunas na ordem padronizada
        Retorna DataFrame com a soma de escolas, matrículas e corpo docente de cada estado / ano"""
        
        lst_tuples = list
        self.df = DataFrame
        
        # Soma os valores das respectivas colunas Matrículas, Escolas, Docentes
        self.df['TOTAL_MATRICULA']   = sum(self.df.iloc[:,lst_tuples[0][0]:lst_tuples[0][1]].transpose().values)
        self.df['TOTAL_ESCOLA']      = sum(self.df.iloc[:,lst_tuples[1][0]:lst_tuples[1][1]].transpose().values)
        self.df['TOTAL_DOCENTE']     = sum(self.df.iloc[:,lst_tuples[2][0]:lst_tuples[2][1]].transpose().values)
        
        return self.df

    def calcular_proporcao(self, list, DataFrame) -> pd.DataFrame:
            
        """
        Recebe um lista de tuplas com os indices de intervalos das colunas e o DataFrame com a estrutura de dados com as colunas na ordem padronizada
        Retorna DataFrame com a proporção de:
        habitantes por inscritos no ENEM,
        habitantes por escolas,
        matrículas por escolas,
        matrículas por docente """
        
        lst_tuples = list
        self.df = DataFrame
        
        # Calcula as proporções dos valores especificados acima
        # self.df['POP/INSC_ENEM']        = (self.df.iloc[:,lst_tuples[0][0]].transpose().values/ \
        #                                              self.df.iloc[:,lst_tuples[0][1]].transpose().values)
            
        self.df['POP/ESCOLA']           = (self.df.iloc[:,lst_tuples[1][0]].transpose().values/ \
                                                      self.df.iloc[:,lst_tuples[1][1]].transpose().values)
            
        self.df['MATRICULA/ESCOLA']     = (self.df.iloc[:,lst_tuples[2][0]].transpose().values/ \
                                                      self.df.iloc[:,lst_tuples[2][1]].transpose().values)
            
        self.df['MATRICULA/DOCENTE']    = (self.df.iloc[:,lst_tuples[3][0]].transpose().values/ \
                                                      self.df.iloc[:,lst_tuples[3][1]].transpose().values)
            
        self.df['DOCENTE/ESCOLA']       = (self.df.iloc[:,lst_tuples[4][0]].transpose().values/ \
                                                      self.df.iloc[:,lst_tuples[4][1]].transpose().values)
        
        return self.df
    
        