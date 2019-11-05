# Lista-1_POS-AI_SD
<b>Lista de exercícios PYTHON</b>
<ol>
  <li>Operadores aritiméticos</li>
  <li>Operadores Relacionais</li>
  <li>Condicionantes</li>
  <li>Listas / Tuplas</li>
  <li>Formatação de strings</li>
 </ol>

# Pré-processamento de dados
<b>Parte 1 - Titanic</b><br />
Nesta primeira parte, vamos focar principalmente nas tarefas de limpeza e transformação de dados. Vamos utilizar novamente os dados dos passageiros do Titanic que estão disponíveis no arquivo titanic_data.csv.<br />

Utilizando a <b>normalização min-max</b>, que é dada pela fórmula:<br />
<b>v′=v−minA / maxA−minA(nmaxA−nminA)+nminA</b><br />
Outra técnica usada foi a <b>discretização</b> dos dados.<br />
Após a discretização dos dados, foi feito o processo de <b>binarização.</b>

<b>Parte 2 - Iris</b><br />
Nesta parte, foi utilizados dados sobre a classificação de plantas Iris a partir de suas características.

Descrição dos Dados: [archive.ics.uci.edu](http://archive.ics.uci.edu/ml/datasets/Iris)

Os dados estarão disponíveis no arquivo iris.csv

Para este exercícios utilizaremos a biblioteca pandas para a leitura e manipulação dos dados. Para o trabalho de redução de dimensionalidade com PCA utilizaremos a implementação disponibilizada pelo Scikit-Learn
