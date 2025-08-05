import pandas as pd
from app.configs.config import DATASET_PATH


def load_data():

   print(DATASET_PATH)
   try:

    dataset = pd.read_csv(DATASET_PATH)

    return dataset

   except :

       print("!! Unable to load dataset")

       return

