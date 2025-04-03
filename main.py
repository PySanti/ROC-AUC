import pandas as pd
from utils.basic_preprocess import basic_preprocess
from utils.generate_tags import generate_tags
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

print("Rendimiento del modelo para conjunto de train")
print(f"Clase negativa : {f1_score(df_train[TARGET], model.predict(df_train.drop(TARGET, axis=1)), pos_label=0):.3f}")
print(f"Clase positiva : {f1_score(df_train[TARGET], model.predict(df_train.drop(TARGET, axis=1)), pos_label=1):.3f}")



print("Rendimiento del modelo para conjunto de test")
print(f"Clase negativa : {f1_score(df_test[TARGET], model.predict(df_test.drop(TARGET, axis=1)), pos_label=0):.3f}")
print(f"Clase positiva : {f1_score(df_test[TARGET], model.predict(df_test.drop(TARGET, axis=1)), pos_label=1):.3f}")

