import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/user/OneDrive/Desktop/Python Bootcamp/22-Deep Learning/DATA/kc_house_data.csv')

# EXPLORATORY DATA ANALYSIS
df.isnull().sum()
df.describe().transpose()

sns.pairplot(df) # correlation between the features
plt.show()

plt.figure(figsize=(12,8))
sns.distplot(df['price'])
plt.show()

sns.countplot(df['bedrooms'])
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='sqft_living',data=df)
plt.show()

sns.boxplot(x='bedrooms',y='price',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='long',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='lat',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=df,hue='price')
plt.show()

df.sort_values('price',ascending=False).head(20)
len(df)*(0.01)
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]

plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',
                data=non_top_1_perc,hue='price',
                palette='RdYlGn',edgecolor=None,alpha=0.2)
plt.show()

sns.boxplot(x='waterfront',y='price',data=df)
plt.show()

#WORKING WITH FEATURE DATA
df.head()
df.info()
df = df.drop('id',axis=1)
df.head()

# FEATURE ENGINEERING FROM DATE
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)

sns.boxplot(x='year',y='price',data=df)
plt.show()
sns.boxplot(x='month',y='price',data=df)
plt.show()


df.groupby('month').mean()['price'].plot(); plt.show()
df.groupby('year').mean()['price'].plot(); plt.show()

df = df.drop('date',axis=1)
df.columns
df['zipcode'].value_counts()
df = df.drop('zipcode',axis=1)
df.head()

# could make sense due to scaling, higher should correlate to more value
df['yr_renovated'].value_counts()
df['sqft_basement'].value_counts()


# SCALING AND TRAIN/TEST SPLIT
X = df.drop('price',axis=1)
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape; X_test.shape

# CREATING A MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# TRAINING THE MODEL
model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400)

losses = pd.DataFrame(model.history.history)
losses.plot(); plt.show()

# EVALUATION ON TEST DATA
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions = model.predict(X_test)
mean_absolute_error(y_test,predictions)
np.sqrt(mean_squared_error(y_test,predictions))
explained_variance_score(y_test,predictions)
df['price'].mean()
df['price'].median()

# Our predictions
plt.scatter(y_test,predictions); plt.show()
# Perfect predictions
plt.plot(y_test,y_test,'r'); plt.show()

errors = y_test.values.reshape(6480, 1) - predictions
sns.distplot(errors); plt.show()

# PREDICTING NEW HOUSE
single_house = df.drop('price',axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1, 19))
model.predict(single_house)
df.iloc[0]