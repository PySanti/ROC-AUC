# ROC-AUC

Este será un proyecto de clasificación en donde el principal objetivo será entender la utilidad de las técnicas de evaluación de algoritmos de aprendizaje supervisado denominadas como curvas `ROC` y `AUC`.

El [dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/discussion/119320) a utilizar contiene información sobre clientes de tarjetas de crédito. La idea básicamente es crear un modelo de aprendizaje supervisado que clasifique en "buenos clientes" y "malos clientes".

Una dificultad de este ejercicio es que, no se incluye un target, por lo tanto, se deben utilizar estrategias de clustering para generarlas.


## Preprocesamiento

Shape del dataset: `(438557, 18)`

Columnas del dataset:
```
ID
CODE_GENDER
FLAG_OWN_CAR
FLAG_OWN_REALTY
CNT_CHILDREN
AMT_INCOME_TOTAL
NAME_INCOME_TYPE
NAME_EDUCATION_TYPE
NAME_FAMILY_STATUS
NAME_HOUSING_TYPE
DAYS_BIRTH
DAYS_EMPLOYED
FLAG_MOBIL
FLAG_WORK_PHONE
FLAG_PHONE
FLAG_EMAIL
OCCUPATION_TYPE
CNT_FAM_MEMBERS
```


### Duplicados 

Si se elimina la columna `ID`, el dataset presenta 348.472 elementos duplicados (79.45%).



### Manejo de nans

La única columna con valores nan es `OCCUPATION_TYPE`.

Contiene 134.203 valores Nan (30.6%), se eliminará la columna.

### Codificación

7 de 18 columnas son categóricas.

La siguiente lista es el conjunto de variables categóricas con sus categorías.

```
CODE_GENDER : 2
FLAG_OWN_CAR : 2
FLAG_OWN_REALTY : 2
NAME_INCOME_TYPE : 5
NAME_EDUCATION_TYPE : 5
NAME_FAMILY_STATUS : 5
NAME_HOUSING_TYPE : 6

```

Con lo anterior, utilizando `OneHotEncoding`, quedaría en un total de 36 columnas (16 - 7 + 27), una cantidad, complemente viable.

### Scalers

Se utilizará `RobustScaler` antes de `PCA`.

### Extracción y selección de características

Se utilizará `PCA` para lidiar con graficacion de datos.

### Outliers

No se estudiará.

### Desequilibrio de datos

Se estudiará después de generar las etiquetas.

### Estudio de distribución gaussiana de los datos

No se realizará.

### Correlaciones

No se estudiarán.

## Generación de etiquetas

Después de ejecutar los pasos de preprocesamiento 1-4 ( mínimos para poder graficar) obtenemos los siguientes resultados.

![Image](./images/1.png)

Después de utilizar `K-MEANS` para la generación de etiquetas, obtuvimos el siguiente resultado.

![Image](./images/2.png)

```
Ratio de varianza de PCA : 0.998
silhouette_score después de utilizar k-means :  0.98
```
(Resultados que parecen demasiado buenos para ser verdad)

## Entrenamiento

## Evaluación

