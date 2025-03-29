from preprocess.encoding import CustomOneHotEncoding
from preprocess.scaler import CustomScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.pipeline import Pipeline
from preprocess.outliers_info import outliers_info




byebye_columns = ["OCCUPATION_TYPE", "ID"]

def basic_preprocess(df):
    pca = PCA(n_components=2)

    df = df.drop(byebye_columns, axis=1)
    df = df.drop_duplicates()

    pipe = Pipeline(steps=[
        ("encoding", CustomOneHotEncoding(df.select_dtypes(include="object").columns.tolist())),
        ("scaler", CustomScaler(df.select_dtypes(exclude="object").columns.tolist())),
        ("pca", pca),
    ])
    df = pd.DataFrame(pipe.fit_transform(df), index=df.index)

    print(f"Ratio de varianza de PCA : {sum(pca.explained_variance_ratio_)}")
    return df
