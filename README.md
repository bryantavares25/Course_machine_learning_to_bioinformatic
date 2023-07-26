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
	Importando bibliotecas >
		import joblib
		import pandas as pd
		from sklearn.model_selection import GridSearchCV
		from sklearn.neural_network import MLPClassifier
		import warnings
		warnings.filterwarnings('ignore', category=FutureWarning)
		warnings.filterwarnings('ignore', category=DeprecationWarning)
		import pandas as pd
		from sklearn.model_selection import train_test_split
	Shannon entropy as a measure of balance >
		On a data set of n instances, if you have k classes of size ci you can compute entropy as follows:
		$H = -\sum_{ i = 1}^k \frac{c_i}{n} \log{ \frac{c_i}{n}}$.
		This is equal to:
		0 when there is one single class. In other words, it tends to 0 when your data set is very unbalanced
		logk when all your classes are balanced of the same size $\frac{n}{k}$
		Therefore, you could use the following measure of Balance for a data set:
		$\mbox{Balance} = \frac{H}{\log{k}} = \frac{-\sum_{ i = 1}^k \frac{c_i}{n} \log{ \frac{c_i}{n}}.  } {\log{k}}$
		which is equal to: 0 for a unbalanced data set | 1 for a balanced data set
	Equilibrar os dados para treino >
		Random sampling >
			Random oversampling involves randomly selecting examples from the minority class, with replacement, and adding them to the training dataset. Random undersampling involves randomly selecting examples from the majority class and deleting them from the training dataset.
			Importando bibliotecas >
				from imblearn.over_sampling import RandomOverSampler
				from imblearn.under_sampling import RandomUnderSampler
		# Synthetic Minority Oversampling Technique
			SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.
			Specifically, a random example from the minority class is first chosen. Then k of the nearest neighbors for that example are found (typically k=5). A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between the two examples in feature space.
		# Combination of SMOTE and Tomek Links Undersampling
			SMOTE is an oversampling method that synthesizes new plausible examples in the minority class.
			Tomek Links refers to a method for identifying pairs of nearest neighbors in a dataset that have different classes. Removing one or both of the examples in these pairs (such as the examples in the majority class) has the effect of making the decision boundary in the training dataset less noisy or ambiguous.
			Specifically, first the SMOTE method is applied to oversample the minority class to a balanced distribution, then examples in Tomek Links from the majority classes are identified and removed.
		Multilayer Perceptron: Fit and evaluate a model > In this section, we will fit and evaluate a simple Multilayer Perceptron model.

### Notebook_05
	Leitura dos dados >
		Importando bibliotecas >
		Matriz de confusão >
	Seleção de características (Feature selection) >
	# Classificador MLP + CV 30/70 sem realizar Balanceamento de Classe
	# Balanceamento de Classe usando SMOTE + Classificador MLP + CV 30/70
	# Univariate feature selection
		A seleção de características univariadas funciona através da seleção das melhores características com base em testes estatísticos univariados. Pode ser visto como uma etapa de pré-processamento para um estimador.
		SelectKBest removes all but the  highest scoring features
		SelectPercentile removes all but a user-specified highest scoring percentage of features using common univariate statistical tests for each feature: false positive rate SelectFpr, false discovery rate SelectFdr, or family wise error SelectFwe.
	# Ranking de importancia de features
	Indução do modelo com as 4 melhores características >
	Indução do modelo com as 2 melhores características >
	
	# Correlation Matrix with Heatmap
	Correlation states how the features are related to each other or the target variable. Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable). Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.
