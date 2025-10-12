import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings


warnings.filterwarnings('ignore')

#importing the dataset
data = pd.read_csv(r"C:\Users\ASUS\Desktop\Machine Leanring\weatherAUS.csv")
df = pd.DataFrame(data)

# Handling missing values
df_cleaned = df.dropna(axis=1, thresh=int(len(df) * 0.25)) #removing columns with more than 75% NaN valeus
df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True)) # fill missing values with mean column values

#encoding
df_encoded = pd.get_dummies(df_cleaned)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#scaling
mm = MinMaxScaler()
columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
#seperating non-categorical columns

X_train[columns] = mm.fit_transform(X_train[columns])
X_test[columns] = mm.transform(X_test[columns])

#exporting files
df_encoded.to_csv('ProcessedWeatherAUS.csv', index=False)
X_train.to_csv('Xtrain.csv', index=False)
X_test.to_csv('Xtest.csv', index=False)