import pandas as pd
from utils.basic_preprocess import basic_preprocess
import pandas as pd
from utils.generate_tags import generate_tags
from utils.show_dataset import show_dataset

df = basic_preprocess(pd.read_csv("./data/data1.csv"))
df = generate_tags(df)
show_dataset(df)


