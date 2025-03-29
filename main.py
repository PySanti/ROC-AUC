import pandas as pd
from utils.basic_preprocess import basic_preprocess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib
from preprocess.outliers_info import outliers_info
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = basic_preprocess(pd.read_csv("./data/data1.csv"))

targets = KMeans(n_clusters=2).fit_predict(df)
print(silhouette_score(df, targets))

df['target'] = targets


matplotlib.use('TkAgg')  # O prueba con 'Qt5Agg'

# Graficar los puntos
plt.scatter(df[df['target'] == 0][0], df[df['target'] == 0][1], color='red', marker='.')
plt.scatter(df[df['target'] == 1][0], df[df['target'] == 1][1], color='blue', marker='.')
plt.title('Gráfico de puntos (Scatter Plot)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Mostrar el gráfico
plt.show()

