# Lista-1_POS-AI_DS
<b>Lista de exercícios PYTHON</b>
<ol>
  <li>Operadores aritiméticos</li>
  <li>Operadores Relacionais</li>
  <li>Condicionantes</li>
  <li>Listas / Tuplas</li>
  <li>Formatação de strings</li>
 </ol>

# Pré-processamento de dados
## Parte 1 - Titanic
Nesta primeira parte, vamos focar principalmente nas tarefas de limpeza e transformação de dados. Vamos utilizar novamente os dados dos passageiros do Titanic que estão disponíveis no arquivo titanic_data.csv.<br />

Utilizando a <b>normalização min-max</b>, que é dada pela fórmula:<br />
<b>v′=v−minA / maxA−minA(ⁿmaxA−ⁿminA)+ⁿminA</b><br />
ⁿ(0,1) => ⁿminA = 0 => ⁿmaxA = 1

Outra técnica usada foi a <b>discretização</b> dos dados.<br />

Após a discretização dos dados, foi feito o processo de <b>binarização.</b>

## Parte 2 - Iris
Nesta parte, foi utilizados dados sobre a classificação de plantas Iris a partir de suas características.

Descrição dos Dados: [archive.ics.uci.edu](http://archive.ics.uci.edu/ml/datasets/Iris)

Os dados estarão disponíveis no arquivo iris.csv

Para este exercícios utilizaremos a biblioteca pandas para a leitura e manipulação dos dados. Para o trabalho de redução de dimensionalidade com PCA utilizaremos a implementação disponibilizada pelo [Scikit-Learn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

# Projeto #1 - Análise Exploratória de Dados

## Introdução

As bases de dados em seu estado puro usualmente não estão prontas para serem trabalhadas. Isso torna a tarefa de explorar dados algo recorrente no dia a dia de um cientista de dados.

Nessa atividade o profissional busca remover dados inconsistentes da base, com valores fora do domínio ou do interesse. Por exemplo removendo registros com valores não preenchidos, ou segmentando a base por um determinado critério. Também se busca entender como esse dado está organizado e como as diversas variáveis existes se relacionam.

## Objetivo

O objetivo desse projeto é realizar uma análise exploratória de dados sobre uma base de interesse, que esteja de acordo com os critérios definidos. Recomendamos a escolha de uma base do [Kaggle](https://www.kaggle.com/) em um tema de interesse ou que você tenha conhecimento sobre, isso facilita muito o processo de investigação.
