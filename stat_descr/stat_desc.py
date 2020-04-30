import numpy as np
import pandas as pd


data_set = pd.read_csv('CardioGoodFitness.csv')

print(data_set.head())

print(data_set.describe(include='all'))