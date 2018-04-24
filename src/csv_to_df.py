import pandas as pd


class DataReader:
    def __init__(self, filepath):
        self.__filepath = filepath
        self.__df = pd.read_csv(filepath)

    def get_df(self):
        return self.__df
