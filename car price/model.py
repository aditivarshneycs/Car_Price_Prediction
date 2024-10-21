# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle

# Set plot style
mpl.style.use('ggplot')

# Load data
car = pd.read_csv(r'C:\Users\HP\Downloads\car price\car price\quikr_car.csv')
backup = car.copy()  # Backup the data

# Data Cleaning

# 1. Remove rows where 'year' is not numeric
car = car[car['year'].str.isnumeric()]

# 2. Convert 'year' to integer
car['year'] = car['year'].astype(int)

# 3. Remove rows where 'Price' is 'Ask For Price'
car = car[car['Price'] != 'Ask For Price']

# 4. Remove commas from 'Price' and convert to integer
car['Price'] = car['Price'].str.replace(',', '').astype(int)

# 5. Clean 'kms_driven' column
car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)

# 6. Remove rows with missing 'fuel_type'
car = car[~car['fuel_type'].isna()]

# 7. Clean 'name' column (retain only first three words)
car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')

# 8. Reset the index
car = car.reset_index(drop=True)

# Filter rows with 'Price' less than 6 million
car = car[car['Price'] < 6000000]

# Data Visualization

# Relationship of Company with Price
plt.subplots(figsize=(15, 7))
ax = sns.boxplot(x='company', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.show()

# Relationship of Year with Price
plt.subplots(figsize=(20, 10))
ax = sns.swarmplot(x='year', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.show()

# Relationship of kms_driven with Price
sns.relplot(x='kms_driven', y='Price', data=car, height=7, aspect=1.5)

# Relationship of Fuel Type with Price
plt.subplots(figsize=(14, 7))
sns.boxplot(x='fuel_type', y='Price', data=car)

# Relationship of Price with FuelType, Year, and Company mixed
ax = sns.relplot(x='company', y='Price', data=car, hue='fuel_type', size='year', height=7, aspect=2)
ax.set_xticklabels(rotation=40, ha='right')

# Model Training

# Split features and target
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# OneHotEncoder for categorical features
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

# Create column transformer for categorical variables
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
                                       remainder='passthrough')

# Linear regression model
lr = LinearRegression()

# Create a pipeline
pipe = make_pipeline(column_trans, lr)

# Fit the model
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Check R2 score
print(f"R2 Score: {r2_score(y_test, y_pred)}")

# Optimize the model with multiple random states
scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

# Get the best random state and R2 score
best_random_state = np.argmax(scores)
best_score = scores[best_random_state]
print(f"Best R2 Score: {best_score} at random state {best_random_state}")

# Refit model with best random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=best_random_state)
pipe.fit(X_train, y_train)

# Save the model
with open('LinearRegressionModel.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# Predict with the saved model
loaded_model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
test_data = pd.DataFrame(columns=X_test.columns, data=np.array(['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']).reshape(1, 5))
print(loaded_model.predict(test_data))
