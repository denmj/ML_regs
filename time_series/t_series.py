import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
import xlrd
from pylab import rcParams

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

df = pd.read_excel("Superstore.xls")
furniture = df.loc[df['Category'] == 'Furniture']


print(df.columns.values)

cols = ['Row ID', 'Order ID',  'Ship Date', 'Ship Mode', 'Customer ID',
 'Customer Name' ,'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region',
 'Product ID', 'Category' ,'Sub-Category', 'Product Name',  'Quantity',
 'Discount', 'Profit']

furniture.drop(cols, axis=1, inplace=True)

# print(furniture.isnull().sum())
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
furniture = furniture.set_index('Order Date')
print(furniture.head())

y = furniture['Sales'].resample('MS').mean()
#
# y.plot(figsize=(12,5))
# plt.show()

#
# rcParams['figure.figsize'] = 15, 6
# decomposition = sm.tsa.seasonal_decompose(y, model='additive')
# fig = decomposition.plot()
# plt.show()


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))