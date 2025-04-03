import pandas as pd
from utils.basic_preprocess import basic_preprocess
from utils.generate_tags import generate_tags
from utils.show_dataset import show_dataset
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# df = basic_preprocess(pd.read_csv("./data/data1.csv"))
# df = generate_tags(df)

TARGET = 'target'

df = pd.read_csv("./data/data_targeted.csv")

df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42, stratify=df[TARGET])

model = BernoulliNB()
model.fit(df_train.drop(TARGET, axis=1), df_train[TARGET])

print("Rendimiento del modelo para conjunto de train")
print(f"Clase negativa : {f1_score(df_train[TARGET], model.predict(df_train.drop(TARGET, axis=1)), pos_label=0)}")
print(f"Clase positiva : {f1_score(df_train[TARGET], model.predict(df_train.drop(TARGET, axis=1)), pos_label=1)}")



print("Rendimiento del modelo para conjunto de test")
print(f"Clase negativa : {f1_score(df_test[TARGET], model.predict(df_test.drop(TARGET, axis=1)), pos_label=0)}")
print(f"Clase positiva : {f1_score(df_test[TARGET], model.predict(df_test.drop(TARGET, axis=1)), pos_label=1)}")

