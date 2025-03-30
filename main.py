import pandas as pd
from utils.basic_preprocess import basic_preprocess
import pandas as pd
from utils.generate_tags import generate_tags
from utils.show_dataset import show_dataset

# df = basic_preprocess(pd.read_csv("./data/data1.csv"))
# df = generate_tags(df)

df = pd.read_csv("./data/data_targeted.csv")
show_dataset(df)
print(df['target'].value_counts())


