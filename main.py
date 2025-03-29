import pandas as pd
from utils.basic_preprocess import basic_preprocess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib

df = basic_preprocess(pd.read_csv("./data/data1.csv"))

matplotlib.use('TkAgg')  # O prueba con 'Qt5Agg'

# Graficar los puntos
plt.scatter(df[0], df[1], color='blue', marker='o')
plt.title('Gráfico de puntos (Scatter Plot)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Mostrar el gráfico
plt.show()

