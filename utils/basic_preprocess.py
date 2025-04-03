from preprocess.encoding import CustomOneHotEncoding
from preprocess.scaler import CustomScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.pipeline import Pipeline
from preprocess.outliers_info import outliers_info
from sklearn.model_selection import train_test_split




byebye_columns = ["OCCUPATION_TYPE", "ID"]

def basic_preprocess(df):

    df = df.drop(byebye_columns, axis=1)
    df = df.drop_duplicates()
    cat_columns = df.select_dtypes(include="object").columns.tolist()
    num_columns = df.select_dtypes(exclude="object").columns.tolist()

    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

    pipe = Pipeline(steps=[
        ("encoding", CustomOneHotEncoding(cat_columns)),
        ("scaler", CustomScaler(num_columns)),
    ])
    df_train = pd.DataFrame(pipe.fit_transform(df_train), index=df_train.index)
    df_test = pd.DataFrame(pipe.transform(df_test), index=df_test.index)

    return [df_train, df_test]
