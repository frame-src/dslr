#!/usr/bin/env python3
from core.training.train import train_model
from core.utils.fileio import save_model
import sys


import pandas as pd 


def load_data(path):
    df = pd.read_csv(path)
    print(df.head)
    print(df.columns)
    return df

def clean_data(df):
    df = df.drop(columns=["First Name", "Last Name", "Birthday", "Best Hand"])
    print(df.iloc[0])

    return df




if __name__ == "__main__":
    df = load_data("data/dataset_train.csv")
    df = clean_data(df)



# if __name__ == "__main__":
#     dataset = 'data/dataset_train.csv'
#     if (len(sys.argv) != 1):
#         print("""The task is to identify the distibution of the data
#                 between all houses in the dataset and find the subject
#                 where it is evenly distributed.""")
#         print("Usage: ./logreg_train.py")
#         sys.exit(1)
#     models = train_model(dataset)
#     if save_model(models) == False:
#         print("Error in saving models.")
#     else:
#         print ("Model saved in ./models/")