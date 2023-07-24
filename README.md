# Course_machine_learning_to_bioinformatic

Atividade prática proposta pelo prof. Márcio Dorn na disciplina de Aprendizado de Máquina para Bioinformática do Programa de Pós-Graduação em Biologia Celular e Molecular, foi proposta a resolução do "Desafio Titanic" (Kaggle competition).

## Desafio Titanic

Desafio: Construção de um modelo que prediga quais passageiros sobrevivêm ao naufrágio.

### Notebooke_01

Conjunto de dados >
	891 passageiros: 
		Name (str) - Name of the passenger
		Pclass (int) - Ticket class
		Sex (str) - Sex of the passenger
		Age (float) - Age in years
		SibSp (int) - Number of siblings and spouses aboard
		Parch (int) - Number of parents and children aboard
		Ticket (str) - Ticket number
		Fare (float) - Passenger fare
		Cabin (str) - Cabin number
		Embarked (str) - Port of embarkation

Leitura dos dados >
	Importando bibliotecas
		import matplotlib.pyplot as plt
		import numpy as np
		import pandas as pd
		import seaborn as sns
	Importando os dados no formato csv como objeto pandas
Limpando variáveis contínuas >
	Idade
	Irmãos e parentes -> family_cnt
Limpando variáveis categóricas >
	Cabine
	Sexo
Criando arquivo com dados limpos > 'titanic_cleaned.csv'

### Notebook_02

Leitura dos dados >
	Importando bibliotecas
		import pandas as pd
		from sklearn.model_selection import train_test_split
		import plotly.express as px
		import plotly.express as px
		from sklearn.decomposition import PCA
	Visualizando os dados de acordo com as características (feat*ures) através de uma 'Scatter matrix' e sobrrevivência como 'Label'
	Redução de dimensionalidade através do PCA (Principal component analysis) de 2 componentes (dá para brincar tirando as caracterísitcas para observar qual tem mais efeito sobre a correlação dos dados)
Fracionamento dos dados entre dados de treino, validação e teste >
	Os dados foram fracionado com o comando 'train_test_split' com 0.6 para treino, 0.2 para validação e 0.2 para teste 
Criando arquivos com os dados fracionados tanto a versão com as características (features) quando com a etiqueta (label)



Notebook_03



Notebook_04



Notebook_05



