# medical-insurance-cost-predication-using-ml

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('medical_insurance.csv')
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
features = ['sex', 'smoker', 'region']

plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
	plt.subplot(1, 3, i + 1)

	x = df[col].value_counts()
	plt.pie(x.values,
			labels=x.index,
			autopct='%1.1f%%')

plt.show()
features = ['sex', 'children', 'smoker', 'region']

plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
	plt.subplot(2, 2, i + 1)
	df.groupby(col).mean()['charges'].plot.bar()
plt.show()
features = ['age', 'bmi']

plt.subplots(figsize=(17, 7))
for i, col in enumerate(features):
	plt.subplot(1, 2, i + 1)
	sb.scatterplot(data=df, x=col,
				y='charges',
				hue='smoker')
plt.show()
features = ['age', 'bmi']

plt.subplots(figsize=(17, 7))
for i, col in enumerate(features):
	plt.subplot(1, 2, i + 1)
	sb.distplot(df[col])
plt.show()
features = ['age', 'bmi']

plt.subplots(figsize=(17, 7))
for i, col in enumerate(features):
	plt.subplot(1, 2, i + 1)
	sb.boxplot(df[col])
plt.show()
df.shape, df[df['bmi']<45].shape
df = df[df['bmi']<45]
for col in df.columns:
	if df[col].dtype == object:
		le = LabelEncoder()
		df[col] = le.fit_transform(df[col])
plt.figure(figsize=(7, 7))
sb.heatmap(df.corr() > 0.8,
		annot=True,
		cbar=False)
plt.show()
features = df.drop('charges', axis=1)
target = df['charges']

X_train, X_val,\
Y_train, Y_val = train_test_split(features, target,
								test_size=0.2,
								random_state=22)
X_train.shape, X_val.shape
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
models = [LinearRegression(), XGBRegressor(),
		RandomForestRegressor(), AdaBoostRegressor(),
		Lasso(), Ridge()]

for i in range(6):
	models[i].fit(X_train, Y_train)

	print(f'{models[i]} : ')
	pred_train = models[i].predict(X_train)
	print('Training Error : ', mape(Y_train, pred_train))

	pred_val = models[i].predict(X_val)
	print('Validation Error : ', mape(Y_val, pred_val))
	print()
