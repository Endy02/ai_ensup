import os
import pandas as pd


class Loader():
    def __init__(self):
        self.data_path = os.path.dirname(os.path.abspath("data")) + "/data/"        
    
    def get_final_dataset(self):
        """
            Load the final dataset
        """
        try:
            final = pd.read_csv(self.data_path + "building-energy-benchmarking-final.csv")
            return final
        except Exception as e:
            raise e
    
    def get_sea_bench_2015(self):
        """
            Load the building energy benchmarking dataset for 2015
        """
        try:
            final = pd.read_csv(self.data_path + "2015-building-energy-benchmarking.csv")
            return final
        except Exception as e:
            raise e
        
    def get_sea_bench_2016(self):
        """
            Load the building energy benchmarking dataset for 2016
        """
        try:
            final = pd.read_csv(self.data_path + "2016-building-energy-benchmarking.csv")
            return final
        except Exception as e:
            raise e