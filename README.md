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
Criando arquivo com dados limpos >
	'titanic_cleaned.csv'

### Notebook_02

Leitura dos dados >
	Importando bibliotecas >
		import pandas as pd
		from sklearn.model_selection import train_test_split
		import plotly.express as px
		import plotly.express as px
		from sklearn.decomposition import PCA
	Visualizando os dados de acordo com as características (feat*ures) através de uma 'Scatter matrix' e sobrrevivência como 'Label'
	Redução de dimensionalidade através do PCA (Principal component analysis) de 2 componentes (dá para brincar tirando as caracterísitcas para observar qual tem mais efeito sobre a correlação dos dados)
Fracionamento dos dados entre dados de treino, validação e teste >
	Os dados foram fracionado com o comando 'train_test_split' com 0.6 para treino, 0.2 para validação e 0.2 para teste 
Criando arquivos com os dados fracionados tanto a versão com as características (features) quando com a etiqueta (label) >
	'train_features.csv'
	'val_features.csv'
	'test_features.csv'
	'train_labels.csv'
	'val_labels.csv'
	'test_labels.csv'

### Notebook_03

Leitura dos dados >
	Importando bibliotecas >
		import joblib import pandas as pd
		from sklearn.model_selection import GridSearchCV
		from sklearn.neural_network import MLPClassifier
		import warnings
		warnings.filterwarnings('ignore', category=FutureWarning)
		warnings.filterwarnings('ignore', category=DeprecationWarning)
		import pandas as pd
		from sklearn.model_selection import train_test_split
PCA com dados de treinamento e PCA com dados de teste >
	Avaliar distribuição e qualidade dos conjuntos selecionados
Matriz de confusão >
Indução do modelo de Multilayer perceptrion >
	Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate.
	Precision = TP/TP+FP
	Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. The question recall answers is: Of all the passengers that truly survived, how many did we label? 
	Recall = TP/TP+FN
	F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.
	F1 Score = 2*(Recall * Precision) / (Recall + Precision)
	Importando bibliotecas >
		from sklearn.neural_network import MLPClassifier
		import warnings
		warnings.filterwarnings('ignore')
		from sklearn.metrics import confusion_matrix
	Busca por hiperparametros utilizando gridsearch >

### Notebook_04

Leitura dos dados >


Notebook_05



