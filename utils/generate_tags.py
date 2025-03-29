from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def generate_tags(df):
    """
        Recibe el dataset preprocesado y utiliza K-MEANS
        para generar las etiquetas
    """
    targets = KMeans(n_clusters=2).fit_predict(df)
    print(f"silhouette_score despues de utilizar k-means : ",silhouette_score(df, targets))
    df['target'] = targets
    return df
