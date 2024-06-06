import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

#conjunto de dados
df = pd.read_csv('mushroom_cleaned.csv')

#todos os dados do CSV parecem relevantes, então vamos deixar todos 

#(o dataset já foi limpo, mas vamos fazer isso por segurança)
# Para atributos contínuos usaremos a média, e para atributos categóricos, a moda
continuous_features = ['cap-diameter', 'stem-height', 'stem-width', 'season']
categorical_features = ['cap-shape', 'gill-attachment', 'gill-color', 'stem-color']

imputer_continuous = SimpleImputer(strategy='mean')
imputer_categorical = SimpleImputer(strategy='most_frequent')

df[continuous_features] = imputer_continuous.fit_transform(df[continuous_features])
df[categorical_features] = imputer_categorical.fit_transform(df[categorical_features])

####discretização de atributos contínuos com KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
df[continuous_features] = discretizer.fit_transform(df[continuous_features])

#70% treinamento, 30% teste
X = df.drop('class', axis=1)  # features
y = df['class']  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
#random_state=42 faz com que a divisão dos dados seja sempre a mesma se usarmos o mesmo valor de random_state

#MLPClassifier
#MLP com 2 camadas ocultas cada uma com 100 neuronios
mlp = MLPClassifier(random_state=1, max_iter=500).fit(X_train, y_train)

# Avaliação do desempenho do modelo no conjunto de teste
accuracy = mlp.score(X_test, y_test)
print("Acurácia do modelo MLP no conjunto de teste:", accuracy)

# Calculando a matriz de confusão
conf_matrix = confusion_matrix(y_test, mlp.predict(X_test))

# Imprimindo a matriz de confusão
print("Matriz de Confusão:")
print(conf_matrix)