import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score  
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
from utils.basic_preprocess import basic_preprocess
from utils.generate_tags import generate_tags
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
from utils.target_unbalance import target_unbalance


TARGET = 'target'

# [df_train, df_test] = basic_preprocess(pd.read_csv("./data/data.csv"))
# df_train = generate_tags(df_train)
# df_test = generate_tags(df_test)

df_train = pd.read_csv("./data/cleaned_data_train.csv")
df_test = pd.read_csv("./data/cleaned_data_test.csv")

target_unbalance(df_train, TARGET)
target_unbalance(df_test, TARGET)

model = BernoulliNB()
model.fit(df_train.drop(TARGET, axis=1), df_train[TARGET])

def find_best_threshold():
    y_scores = model.predict_proba(df_test.drop(TARGET, axis=1))[:, 1]  
    precision, recall, thresholds = precision_recall_curve(df_test[TARGET], y_scores)  
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10) # prevenir division por cero  
    best_index = np.argmax(f1_scores)  
    best_threshold = thresholds[best_index]
    return best_threshold

def model_performance(threshold=None):
    if threshold:
        y_scores = model.predict_proba(df_test.drop(TARGET, axis=1))[:, 1]  
        y_pred_custom = (y_scores >= threshold).astype(int)  
        p_class = f1_score(df_test[TARGET], y_pred_custom, pos_label=1)
        n_class = f1_score(df_test[TARGET], y_pred_custom, pos_label=0)


    else:
        predictions = model.predict(df_test.drop(TARGET, axis=1))
        p_class = f1_score(df_test[TARGET], predictions, pos_label=1)
        n_class = f1_score(df_test[TARGET], predictions, pos_label=0)

    print(f"Clase positiva : {p_class:.3f}")
    print(f"Clase negativa : {n_class:.3f}")

def plot_roc_curve(model, X_test, y_test):  
    # Obtener las probabilidades para la clase positiva  
    y_scores = model.predict_proba(X_test)[:, 1]  
    
    # Calcular la curva ROC  
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)  
    roc_auc = roc_auc_score(y_test, y_scores)  # Área bajo la curva ROC  

    # Graficar la curva ROC  
    plt.figure(figsize=(10, 6))  
    plt.plot(fpr, tpr, color='blue', label='Curva ROC (Área = {:.2f})'.format(roc_auc))  
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Línea diagonal  
    plt.xlim([-2, 2])  
    plt.ylim([-2, 2])  
    plt.xlabel('Tasa de Falsos Positivos')  
    plt.ylabel('Tasa de Verdaderos Positivos')  
    plt.title('Curva ROC')  
    plt.legend(loc='lower right')  
    plt.grid()  
    plt.show()  

def plot_precision_recall_curve(model, X_test, y_test):  
    # Obtener las probabilidades para la clase positiva  
    y_scores = model.predict_proba(X_test)[:, 1]  
    
    # Calcular la curva de precisión-recall  
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)  
    avg_precision = average_precision_score(y_test, y_scores)  # Promedio de precisión  

    # Graficar la curva de precisión-recall  
    plt.figure(figsize=(10, 6))  
    plt.plot(recall, precision, color='blue', label='Curva de Precisión-Recall (Prom. Precisión = {:.2f})'.format(avg_precision))  
    plt.xlim([-2, 2])  
    plt.ylim([-2, 2])  
    plt.xlabel('Recuperación')  
    plt.ylabel('Precisión')  
    plt.title('Curva de Precisión-Recall')  
    plt.legend(loc='lower left')  
    plt.grid()  
    plt.show()  


y_scores = model.predict_proba(df_test.drop(TARGET, axis=1))[:, 1]  

best_thr = find_best_threshold()
print(best_thr)
worst_thr = 0.5

y_pred_custom_best = np.array((y_scores >= best_thr).astype(int))
print(f"Cantidad de 1s para el mejor umbral : {np.sum(y_pred_custom_best==1)}")
print(f"Cantidad de 0s para el mejor umbral : {np.sum(y_pred_custom_best==0)}")

y_pred_custom_worst = np.array((y_scores >= worst_thr).astype(int))
print(f"Cantidad de 1s para el peor umbral : {np.sum(y_pred_custom_worst==1)}")
print(f"Cantidad de 0s para el peor umbral : {np.sum(y_pred_custom_worst==0)}")

y_pred = model.predict(df_test.drop(TARGET, axis=1))
print(f"Cantidad de 1s para el umbral clasico: {np.sum(y_pred==1)}")
print(f"Cantidad de 0s para el umbral clasico: {np.sum(y_pred==0)}")

model_performance(best_thr)
