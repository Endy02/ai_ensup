import numpy as np
import pandas as pd
import os
import json
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns  # for coloring
from sklearn.impute import SimpleImputer

from ai.tools.loader import Loader


class Cleaner():
    def __init__(self):
        self.loader = Loader()
        self.df_2015 = self.loader.get_sea_bench_2015()
        self.df_2016 = self.loader.get_sea_bench_2016()

    def generate_final_dataset(self):
        try:
            df_2015_ = self.df_2015['Location'].map(literal_eval).apply(pd.Series)
            df_2015_2 = df_2015_['human_address'].map(literal_eval).apply(pd.Series)
            df_2015_final = pd.concat([df_2015_.drop(['human_address'], axis=1), df_2015_2], axis=1)
            
            self.df_2015 = pd.concat([self.df_2015.drop(['Location'], axis=1), df_2015_final], axis=1)
            self.df_2015.rename(columns={"latitude": "Latitude",
                                        "longitude": "Longitude",
                                        "address": "Address",
                                        "zip": "ZipCode",
                                        "city": "City",
                                        "state": "State",
                                        "GHGEmissionsIntensity(kgCO2e/ft2)": "GHGEmissionsIntensity",
                                        "GHGEmissions(MetricTonsCO2e)": "TotalGHGEmissions"},
                                inplace=True)
            list_df_2015, list_df_2016 = self.__compare_cols(self.df_2015, self.df_2016)
            
            self.df_2015 = self.df_2015.drop(list_df_2015, axis=1)
            self.df_2016 = self.df_2016.drop(list_df_2016, axis=1)
            
            df_final = pd.concat([self.df_2015, self.df_2016]).reset_index(drop=True)

            df_final.dropna(subset=['TotalGHGEmissions','GHGEmissionsIntensity', 'SiteEUI(kBtu/sf)'])
            # Suppressions des variables qui ne nous intéressent pas
            df_final.drop(['State',
                        'ZipCode',
                        'City', 
                        'Outlier', 
                        'PropertyName', 
                        'TaxParcelIdentificationNumber', 
                        'ComplianceStatus', 
                        'DefaultData',
                        'Address', 
                        'YearsENERGYSTARCertified'], axis=1, inplace=True)

            df_final = self.clean_final_dataset(df_final)

            # Save final dataset
            df_final.to_csv(os.path.dirname(os.path.abspath("data")) + "/data/building-energy-benchmarking-final.csv", index=False)
        except Exception as e:
            raise e

    def clean_final_dataset(self, df_final):
        try:
            # Suppression des variables avec suffixe WN
            componant = []
            for col in df_final.columns:
                if 'WN' in col:
                    componant.append(col)

            df_final.drop(componant, axis=1, inplace=True)

            # Suppression des variables redondantes
            df_final.drop(['ENERGYSTARScore', 'SecondLargestPropertyUseTypeGFA', 'ListOfAllPropertyUseTypes', 'LargestPropertyUseType', 'LargestPropertyUseTypeGFA', 'SecondLargestPropertyUseType', 'NaturalGas(therms)','Electricity(kWh)', 'ThirdLargestPropertyUseTypeGFA', 'ThirdLargestPropertyUseType'], axis=1, inplace=True)

            df_final['Neighborhood'].replace('DELRIDGE NEIGHBORHOODS', 'DELRIDGE', inplace=True)
            df_final['Neighborhood']=df_final['Neighborhood'].map(lambda x: x.upper())

            index_to_drop=df_final[df_final['PropertyGFABuilding(s)']<0].index
            df_final.drop(index_to_drop, inplace=True)

            index_to_drop=df_final[df_final['PropertyGFAParking']<0].index
            df_final.drop(index_to_drop, inplace=True)
            
            df_final['NumberofBuildings'].fillna(0, inplace=True)
            df_final['NumberofBuildings'].replace(0, 1, inplace=True)
            df_final['NumberofFloors'].fillna(0, inplace=True)
            df_final['NumberofFloors'].replace(0, 1, inplace=True)

            columns_to_categorize = ['BuildingType', 'CouncilDistrictCode', 'Neighborhood', 'Latitude', 'Longitude', 'PrimaryPropertyType']
            df_final[columns_to_categorize] = df_final[columns_to_categorize].astype('category')

            df_final = df_final.dropna(subset=['TotalGHGEmissions', 'SiteEUI(kBtu/sf)'])
            df_final = self.__data_structuration(df_final)
            df_final = self.__impute_data(df_final)
        
            return df_final
        except Exception as e:
            raise e

    def quantile_parse(self, df):
        """
            Quantile calculation
            (Not use)        
        """
        try:
            Q1 = np.percentile(df['DIS'], 25, interpolation='midpoint')
            Q3 = np.percentile(df['DIS'], 75, interpolqtion='midpoint')
            IQR = Q3 - Q1
        except Exception as e:
            raise e

    def dataset_core_plot(self, df=None):
        """
            Plot correlation matrix of a dataset
        """
        try:
            if df:
                # Check Correlation
                plt.figure(figsize=(10,10))
                cor = df.corr(numeric_only=True)
                sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
                plt.show()
            else:
                # Check Correlation
                df_final = self.loader.get_final_dataset()
                plt.figure(figsize=(10,10))
                cor = df_final.corr(numeric_only=True)
                sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
                plt.show()
        except Exception as e:
            raise e

    def __data_structuration(self, df_final):
        """
            Correlection feature selections
            :return Dataframe df_new
        """
        try:
            df_final = df_final[~df_final['BuildingType'].str.contains("Multifamily")] #! Select Only non-residential buildings
            target = df_final['TotalGHGEmissions']
            target_2 = df_final['GHGEmissionsIntensity']
            df_before = df_final.drop(['TotalGHGEmissions', 'GHGEmissionsIntensity'], axis=1)
            df_cat = df_before.select_dtypes(include=['category'])
            df_before = df_final.drop(df_cat.columns, axis=1)

            # Delete feature with high correlation
            cor_matrix = df_final.corr(numeric_only=True).abs()
            upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
            df_before = df_before.drop(to_drop, axis=1) # Drop features gretter than 85% of correlation

            df_new = pd.concat([df_before, df_cat, target, target_2], axis=1).reset_index(drop=True)

            return df_new
        except Exception as e:
            raise e

    def __compare_cols(self, df1, df2):
        """
            Compare two dataset and get the list of columns difference
        """
        try:
            df1_idx = list(df1.columns)
            df2_idx = list(df2.columns)

            diff_list_df1 = []
            diff_list_df2 = []

            for item in df1_idx:
                if item not in df2_idx:
                    diff_list_df1.append(item)

            for item in df2_idx:
                if item not in df1_idx:
                    diff_list_df2.append(item)

            return diff_list_df1, diff_list_df2
        except Exception as e:
            raise e

    def __impute_data(self, df_final):
        """
            Return imputed data
            :return SimpleImputer
        """
        try:
            df_qualy = df_final.select_dtypes(include=['category'])
            df_quanty = df_final.select_dtypes(include=['float64','int64'])

            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputation = imputer.fit_transform(df_quanty)
            df_quanty_imp = pd.DataFrame(imputation, columns=df_quanty.columns)

            df_final_new = pd.concat([df_quanty_imp, df_qualy], axis=1)

            return df_final_new
        except Exception as e:
            raise e
