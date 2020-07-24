# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:58:39 2020

@author: TOP Artes
"""

import numpy as np
import pandas as pd

import seaborn as sns

import os
os.environ['PROJ_LIB'] = r'C:\Users\TOP Artes\.conda\pkgs\proj4-5.2.0-ha925a31_1\Library\share'
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# from Orange.evaluation import graph_ranks

sns.set_style('whitegrid')
# %matplotlib inline


class Analise(object):
    
    """
    Classe responsável pelas impressões dos gráficos de análises exploratórias dos dados Estaduais e DF"""
    
    def __init__(self):
        pass


    def compara_dispersao(self, df):        

        df_top = df.sort_values(['MEDIA_NOTAS'], ascending=False).head((500))
        df_top['rank'] = 'TOP'
        df_down = df[df['MEDIA_NOTAS']>0].sort_values(['MEDIA_NOTAS'], ascending=True).head((500))
        df_down['rank'] = 'BOTTOM'
        df_plot = pd.concat([df_top, df_down], axis=0)
        
        
        ax = sns.pairplot(df_plot[['rank','MEDIA_NOTAS','HAB/ESCOLA','MATRIC/ESCOLA',
                                   'DOCENTE/ESCOLA','MATRICULA/DOCENTE']], hue='rank')
        ax.set(ylim=(0, None),xlim=(0, None))
        ax.fig.suptitle('Gráfico de disperção para comparação \n 500 melhores X 500 piores(Média > 0)', y=1.05)             
        ax.fig.text(s='Fonte: INSTITUTO NACIONAL DE ESTUDOS E PESQUISAS EDUCACIONAIS ANÍSIO TEIXEIRA.\n'
'Sinopse Estatística da Educação Básica de 2010 a 2019. Brasília: Inep, 2020.\n'
'Disponível em: <http://portal.inep.gov.br/sinopses-estatisticas-da-educacao-basica>. Acesso em: 05/06/2020\n'

'Fonte: IBGE. Diretoria de Pesquisas - DPE -  Coordenação de População e Indicadores Sociais - COPIS.\n'
'Disponível em: <http://www.dados.gov.br/dataset/cd-censo-demografico>. Acesso em: 05/06/2020\n'
    
'Fonte: IBGE. Diretoria de Pesquisas - DPE -  Coordenação de População e Indicadores Sociais - COPIS.\n'
'Disponível em: <https://www.ibge.gov.br/geociencias/downloads-geociencias.html>. Acesso em: 05/06/2020\n', x=0.01, y=-0.12, fontsize=9, ha='left', va='bottom')
        plt.show()
        
        return
    
    

    def compara_melhores(self, df):
        
        df_top = df.sort_values(['MEDIA_NOTAS'], ascending=False).head((500))
        df_top['rank'] = 'TOP'
        
        municipios = df_top['COD. MUNIC'].head(12).values

        width=0.35
        s = 1
        plt.figure(figsize=(18,50))
        ax = plt.subplot()
        ax.set_ylim(0,25)
        
        for i in range(len(municipios)):
            
            ax = plt.subplot(len(municipios),3,s, sharey=ax, sharex=ax)
            
            if s == 1:        
            
                # Define o local e texto da fonte dos dados
                ax.text(x=1.7, y=1.7, s='Municípios com as 12 melhores médias de 2014 a 2018',
                fontsize=16, ha='center', va='bottom', transform=ax.transAxes)
                    
            
            ax.bar(df[df["COD. MUNIC"]==municipios[i]]['ano'].astype(int) - width/2,
                    df[df["COD. MUNIC"]==municipios[i]]['DOCENTE/ESCOLA'],
                    label='DOCENTE/ESCOLA', width=width)
            
            media = df[df["COD. MUNIC"]==municipios[i]]["MEDIA_NOTAS"].values
            
            for j, p in enumerate(ax.patches):
                ax.annotate(f"Média nts\n {format(media[j], '.2f')}",
                            (p.get_x() + p.get_width(), 25),
                            ha = 'center', va = 'center', xytext = (0, 10),
                            textcoords = 'offset points')
        
                ax.annotate(format(p.get_height(), '.0f'),
                            (p.get_x() + p.get_width() / 2.,p.get_height()),
                            ha = 'center', va = 'center', xytext = (0, 10),
                            textcoords = 'offset points')
            
            
            ax.bar(df[df["COD. MUNIC"]==municipios[i]]['ano'].astype(int) + width/2,
                    df[df["COD. MUNIC"]==municipios[i]]['MATRICULA/DOCENTE'],
                    label='MATRICULA/DOCENTE', width=width)   
        
            for p in ax.patches:
              ax.annotate(format(p.get_height(), '.0f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha = 'center', va = 'center', xytext = (0, 10),
                         textcoords = 'offset points')  
            
            if s == 3:    
                ax.legend()
                # Posiciona a legenda fora do gráfico
                ax.legend(bbox_to_anchor=(.6, 2), loc=2, borderaxespad=0.)
                
            if s == 10:                
                self.get_ref(ax)
                
            municipio = df[df["COD. MUNIC"]==municipios[i]][["NOME_MUNICÍPIO","NM_UF_SIGLA"]].values[0][0] 
            uf = df[df["COD. MUNIC"]==municipios[i]][["NOME_MUNICÍPIO","NM_UF_SIGLA"]].values[0][1] 
            ax.set_title(f'{municipio} / {uf}\nMediana de 2014 a 2018 - {round(np.median(media),2)}\n\n', y=1.05)
            s+=1
        plt.subplots_adjust(hspace=.7, wspace=.1)
        plt.show()
        
        return

        


    def compara_piores(self, df):
        
        
        df_down = df[df['MEDIA_NOTAS']>0].sort_values(['MEDIA_NOTAS'], ascending=True).head((500))
        df_down['rank'] = 'BOTTOM'
 
        municipios = df_down['COD. MUNIC'].head(12).values

        width=0.35
        s = 1
        plt.figure(figsize=(18,50))
        ax = plt.subplot()
        ax.set_ylim(0,25)
        for i in range(len(municipios)):
            
            ax = plt.subplot(len(municipios),3,s, sharey=ax, sharex=ax)  
            
            if s == 1:        
            
                # Define o local e texto da fonte dos dados
                ax.text(x=1.7, y=1.7, s='Municípios com as 12 piores médias de 2014 a 2018',
                fontsize=16, ha='center', va='bottom', transform=ax.transAxes)  
            
            ax.bar(df[df["COD. MUNIC"]==municipios[i]]['ano'].astype(int) - width/2,
                    df[df["COD. MUNIC"]==municipios[i]]['DOCENTE/ESCOLA'],
                    label='DOCENTE/ESCOLA', width=width)
            
            media = df[df["COD. MUNIC"]==municipios[i]]["MEDIA_NOTAS"].values
            
            for j, p in enumerate(ax.patches):
                ax.annotate(f"Média nts\n {format(media[j], '.2f')}",
                            (p.get_x() + p.get_width(), 25),
                            ha = 'center', va = 'center', xytext = (0, 10),
                            textcoords = 'offset points')
        
                ax.annotate(format(p.get_height(), '.0f'),
                            (p.get_x() + p.get_width() / 2.,p.get_height()),
                            ha = 'center', va = 'center', xytext = (0, 10),
                            textcoords = 'offset points')
        
            
            
            ax.bar(df[df["COD. MUNIC"]==municipios[i]]['ano'].astype(int) + width/2,
                    df[df["COD. MUNIC"]==municipios[i]]['MATRICULA/DOCENTE'],
                    label='MATRICULA/DOCENTE', width=width)
        
            for p in ax.patches:
              ax.annotate(format(p.get_height(), '.0f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha = 'center', va = 'center', xytext = (0, 10),
                         textcoords = 'offset points')    
            
            if s == 3:    
                ax.legend()
                # Posiciona a legenda fora do gráfico
                ax.legend(bbox_to_anchor=(.6, 2), loc=2, borderaxespad=0.)                            
                
            if s == 10:                
                self.get_ref(ax)
                
            municipio = df[df["COD. MUNIC"]==municipios[i]][["NOME_MUNICÍPIO","NM_UF_SIGLA"]].values[0][0] 
            uf = df[df["COD. MUNIC"]==municipios[i]][["NOME_MUNICÍPIO","NM_UF_SIGLA"]].values[0][1] 
            ax.set_title(f'{municipio} / {uf}\nMediana de 2014 a 2018 - {round(np.median(media),2)}\n\n', y=1.05)
            s+=1
            
        plt.subplots_adjust(hspace=.7, wspace=.1)
        plt.show()
    
        return
    
    
    
    def compara_correlacao(self, df):
        
        plt.figure(figsize=(20,10))
        ax = plt.axes()
        ax.set_title('\nCorrelação entre as principais variáveis\n\n')
        sns.heatmap(df[df.columns[4:]].corr(), annot=True, ax=ax)
        self.get_ref(ax, -0.12, -.45)
        plt.show()
        
        return
    
    
    
    def compara_docente(self, df):     
        
        df_rank = self.get_rank(df)
    
        medianas_top = df_rank.sort_values(['MEDIA_NOTAS'], ascending=False).head(30)
        
        medianas_down = df_rank.sort_values(['MEDIA_NOTAS'], ascending=True).head(30)
        
        municipios_top = [df[df['COD. MUNIC'] == cod]['NOME_MUNICÍPIO'].values[0] + " / "+ df[df['COD. MUNIC'] == cod]['NM_UF_SIGLA'].values[0]
                          for cod in medianas_top['COD. MUNIC'].values]
        
        municipios_down = [df[df['COD. MUNIC'] == cod]['NOME_MUNICÍPIO'].values[0] + " / "+ df[df['COD. MUNIC'] == cod]['NM_UF_SIGLA'].values[0]
                          for cod in medianas_down['COD. MUNIC'].values]
        width = 0.35
        
        plt.figure(figsize=(20,8))
        ax = plt.subplot(2,1,1)
        
        ax.set_ylim(0,20)
        
        ax.bar(municipios_top, medianas_top['DOCENTE/ESCOLA'], align='edge',
                            label='DOCENTE/ESCOLA', width=width)
        for p in ax.patches:
          ax.annotate(format(p.get_height(), '.0f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center', xytext = (0, 10),
                     textcoords = 'offset points')  
        ax.bar(municipios_top, medianas_top['MATRICULA/DOCENTE'], align='edge',
                            label='MATRICULA/DOCENTE', width=-width)
        for p in ax.patches:
          ax.annotate(format(p.get_height(), '.0f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center', xytext = (0, 10),
                     textcoords = 'offset points')  
        ax.set_title('30 Melhores medianas das médias de 2014 a 2018\n\
        Camparação das medianas das variáveis DOCENTE/ESCOLA e MATRÍCULA/DOCENTE')
        plt.xticks(rotation=60)
        
        ax = plt.subplot(2,1,2, sharey=ax)
        ax.bar(municipios_down, medianas_down['DOCENTE/ESCOLA'], align='edge',
                            label='DOCENTE/ESCOLA', width=width)
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center', xytext = (0, 10),
                       textcoords = 'offset points')  
        ax.bar(municipios_down, medianas_down['MATRICULA/DOCENTE'], align='edge',
                            label='MATRICULA/DOCENTE', width=-width)
        for p in ax.patches:
          ax.annotate(format(p.get_height(), '.0f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center', xytext = (0, 10),
                     textcoords = 'offset points')  
        ax.set_title('30 Piores medianas das médias de 2014 a 2018\n\
        Camparação das medianas das variáveis DOCENTE/ESCOLA e MATRÍCULA/DOCENTE')
        plt.xticks(rotation=60)
        
        ax.legend()
        # Posiciona a legenda fora do gráfico
        ax.legend(bbox_to_anchor=(.9, 3.8), loc=2, borderaxespad=0.)
        self.get_ref(ax, -0.01, -1.85)
        plt.subplots_adjust(hspace=1.5)
        plt.show()
        
        return
    
    def compara_escola(self, df):     
        
        df_rank = self.get_rank(df)
    
        medianas_top = df_rank.sort_values(['MEDIA_NOTAS'], ascending=False).head(30)
        
        medianas_down = df_rank.sort_values(['MEDIA_NOTAS'], ascending=True).head(30)
        
        municipios_top = [df[df['COD. MUNIC'] == cod]['NOME_MUNICÍPIO'].values[0] + " / "+ df[df['COD. MUNIC'] == cod]['NM_UF_SIGLA'].values[0]
                          for cod in medianas_top['COD. MUNIC'].values]
        
        municipios_down = [df[df['COD. MUNIC'] == cod]['NOME_MUNICÍPIO'].values[0] + " / "+ df[df['COD. MUNIC'] == cod]['NM_UF_SIGLA'].values[0]
                          for cod in medianas_down['COD. MUNIC'].values]
        width = 0.35
        
        plt.figure(figsize=(20,8))
        ax = plt.subplot(2,1,1)
        
        ax.set_ylim(0,800)
        
        ax.bar(municipios_top, medianas_top['HAB/ESCOLA'], align='edge',
                            label='HAB/ESCOLA', width=width, color='r')
        for p in ax.patches:
          ax.annotate(format(p.get_height(), '.0f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center', xytext = (0, 10),
                     textcoords = 'offset points')  
        ax.bar(municipios_top, medianas_top['MATRIC/ESCOLA'], align='edge',
                            label='MATRIC/ESCOLA', width=-width, color='g')
        for p in ax.patches:
          ax.annotate(format(p.get_height(), '.0f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center', xytext = (0, 10),
                     textcoords = 'offset points')  
        ax.set_title('30 Melhores medianas das médias de 2014 a 2018\n\
        Camparação das medianas das variáveis HAB/ESCOLA e MATRÍCULA/ESCOLA')
        plt.xticks(rotation=60)
        
        ax = plt.subplot(2,1,2, sharey=ax)
        ax.bar(municipios_down, medianas_down['HAB/ESCOLA'], align='edge',
                            label='HAB/ESCOLA', width=width, color='r')
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center', xytext = (0, 10),
                       textcoords = 'offset points')  
        ax.bar(municipios_down, medianas_down['MATRIC/ESCOLA'], align='edge',
                            label='MATRIC/ESCOLA', width=-width, color='g')
        for p in ax.patches:
          ax.annotate(format(p.get_height(), '.0f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center', xytext = (0, 10),
                     textcoords = 'offset points')  
        ax.set_title('30 Piores medianas das médias de 2014 a 2018\n\
        Camparação das medianas das variáveis HAB/ESCOLA e MATRÍCULA/ESCOLA')
        plt.xticks(rotation=60)
        
        ax.legend()
        # Posiciona a legenda fora do gráfico
        ax.legend(bbox_to_anchor=(.9, 3.8), loc=2, borderaxespad=0.)
        self.get_ref(ax, -0.01, -1.85)
        plt.subplots_adjust(hspace=1.5)
        plt.show()
        
        return
    
    
    def compara_densidade(self, df):       
        
        df_rank = self.get_rank(df)

        
        locations = pd.read_csv('CSVs/localidadesBR_LON_LAT.csv', sep=';', decimal=',')
        locations.columns = ['COD. MUNIC','longd','latd']
        df_rank = pd.merge(df_rank,locations, on='COD. MUNIC')
        
        
        # Extract the data we're interested in
        lat = df_rank['latd'].values
        lon = df_rank['longd'].values
        matriculas = df_rank['MATRIC/ESCOLA'].values
        escolas = df_rank['HAB/ESCOLA'].values
        matriculas_doc = df_rank['MATRICULA/DOCENTE'].values
        docentes_esc = df_rank['DOCENTE/ESCOLA'].values
        medias = df_rank['MEDIA_NOTAS'].values
        
        
        # 1. Draw the map background
        fig = plt.figure(figsize=(40, 40))
        
        ax = fig.add_subplot(311)        
        ax.set_title('Densidade de Docentes / Escolas e Matrículas / Docentes') 
        
        m = Basemap(projection='lcc', resolution='l', 
                    lat_0=-12, lon_0=-55,
                    width=5E6, height=5.2E6)
        m.shadedrelief()
        m.drawcoastlines(color='gray')
        m.drawcountries(color='gray')
        m.drawstates(color='gray')
        
        # 2. scatter city data, with color reflecting population
        # and size reflecting area
        m.scatter(lon, lat, latlon=True,
                  c=docentes_esc, s=matriculas_doc,
                  cmap='Oranges', alpha=0.7)
        
        # 3. create colorbar and legend
        plt.colorbar(label=r'Docentes / Escolas')
        plt.clim(-10, max(matriculas_doc))
        
        # make legend with dummy points
        for a in [int(min(docentes_esc)), int(np.median(docentes_esc)), int(max(docentes_esc))]:
            plt.scatter([], [], c='k', alpha=0.5, s=a,
                        label=str(a) + ' Matrículas / Docentes')
            
        plt.legend(scatterpoints=1, frameon=True,
                   labelspacing=1, loc='lower left');
        
        self.get_annotate(m, df, df_rank)
        
        ax = fig.add_subplot(312)        
        ax.set_title('Densidade de Habitantes e Matrículas / Escolas')
        
        m = Basemap(projection='lcc', resolution='l', 
                    lat_0=-12, lon_0=-55,
                    width=5E6, height=5.2E6)
        m.shadedrelief()
        m.drawcoastlines(color='gray')
        m.drawcountries(color='gray')
        m.drawstates(color='gray')
        
        # 2. scatter city data, with color reflecting Matrículas / Escolas
        # and size reflecting Escolas / Habitantes
        m.scatter(lon, lat, latlon=True,
                  c=matriculas, s=escolas/matriculas*1.7,
                  cmap='Reds', alpha=0.7)
        
        # 3. create colorbar and legend
        plt.colorbar(label=r'Matrículas / Escolas')
        plt.clim(min(matriculas)-10, max(matriculas))
        
        # make legend with dummy points
        for a in [int(min(escolas)), int(np.median(escolas)), int(max(escolas))]:
            plt.scatter([], [], c='k', alpha=0.5, s=np.log10(a)*5,
                        label=str(a) + ' Habitantes / Escolas')
        plt.legend(scatterpoints=1, frameon=True,
                   labelspacing=1, loc='lower left');
        plt.subplots_adjust(hspace=.07)
        
        self.get_annotate(m, df, df_rank)
        
        ax = fig.add_subplot(313)        
        ax.set_title('Densidade das medianas de 2014 a 2018') 
        
        m = Basemap(projection='lcc', resolution='l', 
                    lat_0=-12, lon_0=-55,
                    width=5E6, height=5.2E6)
        m.shadedrelief()
        m.drawcoastlines(color='gray')
        m.drawcountries(color='gray')
        m.drawstates(color='gray')
        
        # 2. scatter city data, with color reflecting population
        # and size reflecting area
        m.scatter(lon, lat, latlon=True,
                  c=medias, s=np.log2(medias),
                  cmap='inferno', alpha=0.7)
        
        # 3. create colorbar and legend
        plt.colorbar(label=r'Meidana das médias (2014 a 2018)')
        plt.clim(0, max(medias))
        
        # make legend with dummy points
        for a in [int(min(medias)), int(np.median(medias)), int(max(medias))]:
            plt.scatter([], [], c='k', alpha=0.5, s=np.log2(a),
                        label=str(a) + ' Medianas das Médias')
            
        plt.legend(scatterpoints=1, frameon=True,
                   labelspacing=1, loc='lower left');
        
        self.get_annotate(m, df, df_rank)
        self.get_ref(ax,0.02,-0.15)
        
        plt.show()
        
        return
        
    def get_rank(self, df):        

        df_rank = df.groupby('COD. MUNIC')[['HAB/ESCOLA','MATRIC/ESCOLA',
                                               'DOCENTE/ESCOLA','MATRICULA/DOCENTE',
                                               'MEDIA_NOTAS']].median().reset_index()
        
        return df_rank
    
    
    def get_outliers(self, df):
        
        plt.figure(figsize=(20,15))

        plt.subplot(211)
        sns.boxplot(x='ano', y='MEDIA_NOTAS', data=df)
        plt.title('Detecção dos "outliiers" de cada ano')
        
        #plt.figure(figsize=(20,20))
        plt.subplot(212)
        ax = sns.boxplot(x='ano', y='MEDIA_NOTAS', data=df, hue='CLASSE')
        
        plt.title('Detecção dos "outliiers" de cada ano separados por classe')
        
        labels=['Insatisfatório', 'Regular', 'Satisfatório']
        h,l = ax.get_legend_handles_labels()
        ax.legend(handles=h, labels=labels)
        
        self.get_ref(ax, -0.03, -0.4)
        
        plt.show()
        
        return

    
    def get_annotate(self ,m, df, df_rank):
        
        plt.annotate(f'Melhores',
                     xy=m(-65, 8),  xycoords='data',
                        xytext=m(-73, 8), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='g'),
                        bbox=dict(boxstyle="round", fc="w"))
        
        plt.annotate(f'Piores    ',
                     xy=m(-65, 6),  xycoords='data',
                        xytext=m(-73, 6), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='r'),
                        bbox=dict(boxstyle="round", fc="w"))
        
        medianas_top = df_rank.sort_values(['MEDIA_NOTAS'], ascending=False).head(5)
    
        x, y =  m(medianas_top.iloc[0]['longd'], medianas_top.iloc[0]['latd'])
        x2, y2 = m(medianas_top.iloc[0]['longd']+(5), medianas_top.iloc[0]['latd']-5)
        
        plt.annotate(f'{df[df["COD. MUNIC"]==medianas_top.iloc[0]["COD. MUNIC"]]["NOME_MUNICÍPIO"].values[0]} - {df[df["COD. MUNIC"]==medianas_top.iloc[0]["COD. MUNIC"]]["NM_UF_SIGLA"].values[0]}\nMediana {round(medianas_top.iloc[0]["MEDIA_NOTAS"],2)}',
                     xy=(x, y),  xycoords='data',
                        xytext=(x2, y2), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='g'))
        
        x, y =  m(medianas_top.iloc[1]['longd'], medianas_top.iloc[1]['latd'])
        x2, y2 = m(medianas_top.iloc[1]['longd']+(5), medianas_top.iloc[1]['latd'])
        
        plt.annotate(f'{df[df["COD. MUNIC"]==medianas_top.iloc[1]["COD. MUNIC"]]["NOME_MUNICÍPIO"].values[0]} - {df[df["COD. MUNIC"]==medianas_top.iloc[1]["COD. MUNIC"]]["NM_UF_SIGLA"].values[0]}\nMediana {round(medianas_top.iloc[1]["MEDIA_NOTAS"],2)}',
                     xy=(x, y),  xycoords='data',
                        xytext=(x2, y2), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='g'))
        
        x, y =  m(medianas_top.iloc[2]['longd'], medianas_top.iloc[2]['latd'])
        x2, y2 = m(medianas_top.iloc[2]['longd']+(8), medianas_top.iloc[2]['latd']+2)
        
        plt.annotate(f'{df[df["COD. MUNIC"]==medianas_top.iloc[2]["COD. MUNIC"]]["NOME_MUNICÍPIO"].values[0]} - {df[df["COD. MUNIC"]==medianas_top.iloc[2]["COD. MUNIC"]]["NM_UF_SIGLA"].values[0]}\nMediana {round(medianas_top.iloc[2]["MEDIA_NOTAS"],2)}',
                     xy=(x, y),  xycoords='data',
                        xytext=(x2, y2), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='g'))
        
        x, y =  m(medianas_top.iloc[3]['longd'], medianas_top.iloc[3]['latd'])
        x2, y2 = m(medianas_top.iloc[3]['longd']+(5), medianas_top.iloc[3]['latd']-3)
        
        plt.annotate(f'{df[df["COD. MUNIC"]==medianas_top.iloc[3]["COD. MUNIC"]]["NOME_MUNICÍPIO"].values[3]} - {df[df["COD. MUNIC"]==medianas_top.iloc[3]["COD. MUNIC"]]["NM_UF_SIGLA"].values[3]}\nMediana {round(medianas_top.iloc[3]["MEDIA_NOTAS"],2)}',
                     xy=(x, y),  xycoords='data',
                        xytext=(x2, y2), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='g'))
        
        x, y =  m(medianas_top.iloc[4]['longd'], medianas_top.iloc[4]['latd'])
        x2, y2 = m(medianas_top.iloc[4]['longd']+(5), medianas_top.iloc[4]['latd']-4)
        
        plt.annotate(f'{df[df["COD. MUNIC"]==medianas_top.iloc[4]["COD. MUNIC"]]["NOME_MUNICÍPIO"].values[0]} - {df[df["COD. MUNIC"]==medianas_top.iloc[4]["COD. MUNIC"]]["NM_UF_SIGLA"].values[0]}\nMediana {round(medianas_top.iloc[4]["MEDIA_NOTAS"],2)}',
                     xy=(x, y),  xycoords='data',
                        xytext=(x2, y2), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='g'))
        
        
        
        medianas_down = df_rank.sort_values(['MEDIA_NOTAS'], ascending=True).head(5)
    
        x, y =  m(medianas_down.iloc[0]['longd'], medianas_down.iloc[0]['latd'])
        x2, y2 = m(medianas_down.iloc[0]['longd']-7, medianas_down.iloc[0]['latd']+28)
        
        plt.annotate(f'{df[df["COD. MUNIC"]==medianas_down.iloc[0]["COD. MUNIC"]]["NOME_MUNICÍPIO"].values[0]} - {df[df["COD. MUNIC"]==medianas_down.iloc[0]["COD. MUNIC"]]["NM_UF_SIGLA"].values[0]}\nMediana {round(medianas_down.iloc[0]["MEDIA_NOTAS"],2)}',
                     xy=(x, y),  xycoords='data',
                        xytext=(x2, y2), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='r'))
    
        x, y =  m(medianas_down.iloc[1]['longd'], medianas_down.iloc[1]['latd'])
        x2, y2 = m(medianas_down.iloc[1]['longd']+2, medianas_down.iloc[1]['latd']+20)
        
        plt.annotate(f'{df[df["COD. MUNIC"]==medianas_down.iloc[1]["COD. MUNIC"]]["NOME_MUNICÍPIO"].values[0]} - {df[df["COD. MUNIC"]==medianas_down.iloc[1]["COD. MUNIC"]]["NM_UF_SIGLA"].values[0]}\nMediana {round(medianas_down.iloc[1]["MEDIA_NOTAS"],2)}',
                     xy=(x, y),  xycoords='data',
                        xytext=(x2, y2), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='r')) 
        
        x, y =  m(medianas_down.iloc[2]['longd'], medianas_down.iloc[2]['latd'])
        x2, y2 = m(medianas_down.iloc[2]['longd']+2, medianas_down.iloc[2]['latd']+12)
        
        plt.annotate(f'{df[df["COD. MUNIC"]==medianas_down.iloc[2]["COD. MUNIC"]]["NOME_MUNICÍPIO"].values[0]} - {df[df["COD. MUNIC"]==medianas_down.iloc[2]["COD. MUNIC"]]["NM_UF_SIGLA"].values[0]}\nMediana {round(medianas_down.iloc[2]["MEDIA_NOTAS"],2)}',
                     xy=(x, y),  xycoords='data',
                        xytext=(x2, y2), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='r'))
        
        x, y =  m(medianas_down.iloc[3]['longd'], medianas_down.iloc[3]['latd'])
        x2, y2 = m(medianas_down.iloc[3]['longd']+11, medianas_down.iloc[3]['latd']+26)
        
        plt.annotate(f'{df[df["COD. MUNIC"]==medianas_down.iloc[3]["COD. MUNIC"]]["NOME_MUNICÍPIO"].values[0]} - {df[df["COD. MUNIC"]==medianas_down.iloc[3]["COD. MUNIC"]]["NM_UF_SIGLA"].values[0]}\nMediana {round(medianas_down.iloc[3]["MEDIA_NOTAS"],2)}',
                     xy=(x, y),  xycoords='data',
                        xytext=(x2, y2), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='r'))
        
        x, y =  m(medianas_down.iloc[4]['longd'], medianas_down.iloc[4]['latd'])
        x2, y2 = m(medianas_down.iloc[4]['longd']-1, medianas_down.iloc[4]['latd']+12)
        
        plt.annotate(f'{df[df["COD. MUNIC"]==medianas_down.iloc[4]["COD. MUNIC"]]["NOME_MUNICÍPIO"].values[0]} - {df[df["COD. MUNIC"]==medianas_down.iloc[4]["COD. MUNIC"]]["NM_UF_SIGLA"].values[0]}\nMediana {round(medianas_down.iloc[4]["MEDIA_NOTAS"],2)}',
                     xy=(x, y),  xycoords='data',
                        xytext=(x2, y2), textcoords='data',
                        color='k',size = 'small',
                        arrowprops=dict(arrowstyle="->", color='r'))
        
        return

    def get_ref(self, ax, x=0, y=0):
        
        x = -0.03 if x == 0 else x
        y = -0.85 if y == 0 else y
        
                # Define o local e texto da fonte dos dados
        ax.text(x=x, y=y, s='Fonte: INSTITUTO NACIONAL DE ESTUDOS E PESQUISAS EDUCACIONAIS ANÍSIO TEIXEIRA.\n'
'Sinopse Estatística da Educação Básica de 2010 a 2019. Brasília: Inep, 2020.\n'
'Disponível em: <http://portal.inep.gov.br/sinopses-estatisticas-da-educacao-basica>. Acesso em: 05/06/2020\n'

'Fonte: IBGE. Diretoria de Pesquisas - DPE -  Coordenação de População e Indicadores Sociais - COPIS.\n'
'Disponível em: <http://www.dados.gov.br/dataset/cd-censo-demografico>. Acesso em: 05/06/2020\n'
    
'Fonte: IBGE. Diretoria de Pesquisas - DPE -  Coordenação de População e Indicadores Sociais - COPIS.\n'
'Disponível em: <https://www.ibge.gov.br/geociencias/downloads-geociencias.html>. Acesso em: 05/06/2020\n',
    fontsize=9, ha='left', va='bottom', transform=ax.transAxes)
        
        return