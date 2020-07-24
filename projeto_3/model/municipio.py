# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:57:38 2020

@author: TOP Artes
"""
# Importa a biblioteca para estruturação dos dados e visualização dos dados   
import pandas as pd

class Municipio(object):
    
    def __init__(self):
        self.df_raw = pd.DataFrame()   
        self.df = pd.DataFrame()      
        self.lst_anos = list([2014,2015,2016,2017,2018])
     
    def gerar_dados(self) -> pd.DataFrame:
        
        """
        Recebe a lista com os anos a serem concatenados no dataframe
        Retorna o DataFrame de desempenho(MEDIANA) no ENEM de cada estado """
        
        # Carrega dos dados das tabelas de 2014 a 2019        
        for ano in self.lst_anos:
            
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
            df_pop_ano.drop(['UF','COD. UF','COD_MUNIC','NOME DO MUNICÍPIO'], axis=1, inplace=True)
            
            df_territorio_ano = pd.merge(df_territorio_ano[['ano','NM_UF_SIGLA','COD. MUNIC','NOME_MUNICÍPIO','AREA_km²']], df_pop_ano, on='COD. MUNIC')           
            
            df_educacao_ano = pd.read_csv(f'CSVs/{ano}_educacao_basica.csv', sep=';', decimal=',')            
            df_tmp = pd.merge(df_territorio_ano, df_educacao_ano, on='COD. MUNIC')
            
            df_enem = pd.read_csv(f'CSVs/{ano}_MICRODADOS_ENEM_500K.csv', sep=';')
            
            if ano == 2014:
                df_enem1                    = df_enem[['COD_MUNICIPIO_RESIDENCIA','NOTA_CN', 'NOTA_CH', 'NOTA_LC', 'NOTA_MT','NU_NOTA_REDACAO']].groupby('COD_MUNICIPIO_RESIDENCIA').median()
                df_enem1['CAND_ESC_PUB']    = df_enem[df_enem['TP_ESCOLA']==1].groupby('COD_MUNICIPIO_RESIDENCIA')['TP_ESCOLA'].count()
                df_enem1['CAND_ESC_PRI']    = df_enem[df_enem['TP_ESCOLA']==2].groupby('COD_MUNICIPIO_RESIDENCIA')['TP_ESCOLA'].count()               

                df_enem1['ENS_REGULAR']     = df_enem[df_enem['IN_TP_ENSINO']==1].groupby('COD_MUNICIPIO_RESIDENCIA')['IN_TP_ENSINO'].count()
                df_enem1['ENS_EJA']         = df_enem[df_enem['IN_TP_ENSINO']==2].groupby('COD_MUNICIPIO_RESIDENCIA')['IN_TP_ENSINO'].count()
                df_enem1['ENS_ESPECIAL']    = df_enem[df_enem['IN_TP_ENSINO']==4].groupby('COD_MUNICIPIO_RESIDENCIA')['IN_TP_ENSINO'].count()
                df_enem1 = df_enem1.fillna(0)
                
            else:
                df_enem1                    = df_enem[['CO_MUNICIPIO_RESIDENCIA','NU_NOTA_CN','NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT','NU_NOTA_REDACAO']].groupby('CO_MUNICIPIO_RESIDENCIA').median()
                
                df_enem1['CAND_ESC_PUB']    = df_enem[df_enem['TP_ESCOLA']==2].groupby('CO_MUNICIPIO_RESIDENCIA')['TP_ESCOLA'].count()
                df_enem1['CAND_ESC_PRI']    = df_enem[df_enem['TP_ESCOLA']==3].groupby('CO_MUNICIPIO_RESIDENCIA')['TP_ESCOLA'].count()               

                df_enem1['ENS_REGULAR']     = df_enem[df_enem['TP_ENSINO']==1].groupby('CO_MUNICIPIO_RESIDENCIA')['TP_ENSINO'].count()
                df_enem1['ENS_EJA']         = df_enem[df_enem['TP_ENSINO']==2].groupby('CO_MUNICIPIO_RESIDENCIA')['TP_ENSINO'].count()
                df_enem1['ENS_ESPECIAL']    = df_enem[df_enem['TP_ENSINO']==4].groupby('CO_MUNICIPIO_RESIDENCIA')['TP_ENSINO'].count()
                df_enem1 = df_enem1.fillna(0)

            df_enem = df_enem1.reset_index()
            df_enem.columns = ['COD. MUNIC','NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC',
                               'NU_NOTA_MT','NU_NOTA_REDACAO','CAND_ESC_PUB','CAND_ESC_PRI',
                               'ENS_REGULAR','ENS_EJA','ENS_ESPECIAL']
            
            df_tmp = pd.merge(df_tmp, df_enem, on='COD. MUNIC')
            df_tmp = df_tmp.fillna(0)
            self.df_raw = pd.concat([self.df_raw, df_tmp], ignore_index=True)
        
        return self.df_raw
 
    
    def tratar_dados(self, list) -> pd.DataFrame:
        
        """
        Recebe a lista com as colunas e o DataaFrame de dados a serem tratados e convertidos
        Retorna o DataFrame com os dados tratados -> NaN = 0 / StringObj = Int """ 
        
        lst_colunas = list
        self.df = self.df_raw.copy()
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
        
    def calcular_densidade(self) -> pd.DataFrame:
        
        """
        Recebe um lista de tuplas com os indices de intervalos das colunas e o DataFrame com a estrutura de dados com as colunas na ordem padronizada
        Retorna DataFrame com a soma de escolas, matrículas e corpo docente de cada estado / ano """

        
        # Calcula a densidade de população, escolas, matrículas e docentes por km² por km² de cada estado 
        # Estimativa de habitante/km²
        self.df['HAB/km²']                   = (self.df.iloc[:,5].transpose().values/ \
                                                           self.df.iloc[:,4].transpose().values)
        
        # Estimativa de Escolas/km²
        self.df['ESCOLA/km²']                = (self.df.iloc[:,63].values/ \
                                                           self.df.iloc[:,4].values)
        
        # Estimativa de habitantes(km²)/escolas(km²)df_raw
        self.df['HAB/ESCOLA']                = (self.df.iloc[:,5].values/ \
                                                           self.df.iloc[:,63].values)
         
        # Estimativa de matrículas(km²)/escolas(km²)
        self.df['MATRIC/ESCOLA']            = (self.df.iloc[:,60].values/ \
                                                            self.df.iloc[:,63].values)
         
        # Estimativa do corpo docente(km²)/escolas(km²)
        self.df['DOCENTE/ESCOLA']           = (self.df.iloc[:,64].values/ \
                                                           self.df.iloc[:,63].values)
        
        # Estimativa do corpo docente(km²)/matrículas(km²)
        self.df['MATRICULA/DOCENTE']        = (self.df.iloc[:,60].values/ \
                                                            self.df.iloc[:,64].values)
        
        return self.df
    
    
    
    def calcular_totais(self) -> pd.DataFrame:
            
        """
        Recebe um lista de tuplas com os indices de intervalos das colunas e o DataFrame com a estrutura de dados com as colunas na ordem padronizada
        Retorna DataFrame com a soma de escolas, matrículas e corpo docente de cada estado / ano"""
        
        self.df['MEDIA_NOTAS']            = self.df.iloc[:,49:54].transpose().mean()
        
        # Soma os valores das respectivas colunas Matrículas, Escolas, Docentes
        self.df['TOTAL_MATRICULA']       = sum(self.df.iloc[:,6:13].transpose().values)
        self.df['TOTAL_MATRICULA_PUB']   = sum(self.df.iloc[:,[27,28,29,31,32,33]].transpose().values)
        self.df['TOTAL_MATRICULA_PRI']   = sum(self.df.iloc[:,[30,34]].transpose().values)
        self.df['TOTAL_ESCOLA']          = sum(self.df.iloc[:,13:20].transpose().values)
        self.df['TOTAL_DOCENTE']         = sum(self.df.iloc[:,20:27].transpose().values)
        self.df['TOTAL_DOCENTE_PUB']     = sum(self.df.iloc[:,[35,37]].transpose().values)
        self.df['TOTAL_DOCENTE_PRI']     = sum(self.df.iloc[:,[36,38]].transpose().values)
        self.df['TOTAL_ESCOLA_PUB']      = sum(self.df.iloc[:,[42,43,45,46]].transpose().values)
        self.df['TOTAL_ESCOLA_PRI']      = sum(self.df.iloc[:,[44,48]].transpose().values)
        
        return self.df
    
    
    def classificar(self):
        
        self.df.loc[self.df['MEDIA_NOTAS'] >= 600, 'CLASSE'] = 2         # Classifica os resultados satisfatórios como 2
        self.df.loc[self.df['MEDIA_NOTAS'] < 600, 'CLASSE'] = 1          # Classifica os resultados regulares como 1
        self.df.loc[self.df['MEDIA_NOTAS'] < 450, 'CLASSE'] = 0         # Classifica os resultados insatisfatórios como 0
        
        return self.df
        