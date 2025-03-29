# ROC-AUC

Este sera un proyecto de clasificacion en donde el principal objetivo sera entender la utilidad de las tecnicas de evaluacion de algoritmos de aprendizaje supervisado denominadas como curvas `ROC` y `AUC`.

El [dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/discussion/119320) a utilizar contiene informacion sobre clientes de tarjetas de credito. La idea basicamente es crear un modelo de aprendizaje supervisado que clasifique en "buenos clientes" y "malos clientes".

Una dificultad de este ejercicio es que, no se incluye un target, por lo tanto, se deben utilizar estrategias de aprendizaje no supervisado para generarlas.



## Preprocesamiento

1- Manejo de nans: la unica columna con valores nan es `COCUPATION_TYPE`.

Contiene 134.203 valores Nan (30.6%), se eliminara la columna.

2- Codificacion: 8 de 18 columnas son categoricas ...

La siguiente lista es el conjunto de variables categoricas con sus categorias diferentes.

```
CODE_GENDER : 2
FLAG_OWN_CAR : 2
FLAG_OWN_REALTY : 2
NAME_INCOME_TYPE : 5
NAME_EDUCATION_TYPE : 5
NAME_FAMILY_STATUS : 5
NAME_HOUSING_TYPE : 6

```

Con lo anterior, utilizando `OneHotEncoding`, quedaria en un total de 55 columnas (18 - 8 + 45)

3- Scalers:

4- Extraccion y seleccion de caracteristicas:

5- Outliers:

6- Desequilibrio de datos:

7- Estudio de distribucion gaussiana de los datos:

8- Correlaciones:

## Generacion de etiquetas

## Entrenamiento

## Evaluacion
