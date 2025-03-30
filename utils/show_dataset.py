import matplotlib.pyplot as plt
import matplotlib

def show_dataset(df):
    """
        Genera un grafico mostrando las clases del dataset
    """
    matplotlib.use('TkAgg')

    plt.scatter(df[df['target'] == 0]['0'], df[df['target'] == 0]['1'], color='red', marker='.')
    plt.scatter(df[df['target'] == 1]['0'], df[df['target'] == 1]['1'], color='blue', marker='.')
    plt.title('Gráfico de puntos (Scatter Plot)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.show()

