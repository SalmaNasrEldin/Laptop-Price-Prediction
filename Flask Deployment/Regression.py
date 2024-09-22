#importing the libraries
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt

#reading the csv file into a dataframe
df=pd.read_csv('ElecDeviceRatingPrediction.csv')


df['processor_gnrtn'] = df['processor_gnrtn'].replace('Not Available', np.nan)
"""Removing 'th' in the processor_gnrtn column and turning it into a float column"""
# Function to convert the generation to float and maintain null values
def convert_to_float(gnrtn):
    if pd.isnull(gnrtn):
        return gnrtn
    else:
        return int(gnrtn.rstrip('th'))
# Apply the function to the 'processor_gnrtn' column
df['processor_gnrtn'] = df['processor_gnrtn'].apply(convert_to_float)




"""##Changing all features that have string extension to number only"""
df['rating'] = df['rating'].replace({' stars':'', 'star':''}, regex=True)
df['rating'] =pd.to_numeric(df['rating'])
df = df.replace(' GB', '', regex=True)
df['ram_gb'] = pd.to_numeric(df['ram_gb'])
df['ssd'] =pd.to_numeric(df['ssd'])
df['hdd'] =pd.to_numeric(df['hdd'])
df['graphic_card_gb'] =pd.to_numeric(df['graphic_card_gb'])



"""##One-hot encoder and label encoder for the processor_name column"""
#processor name: [core 3, 5, 7, 9] [ryzen 3, 5, 7, 9] [pentQuad] [celeron] [m1] --> 5 new col
df['Pentium_Quad'] = [1 if x == 'Pentium Quad' else 0 for x in df['processor_name']] #entry = 1 if df['processor_name']== 'Pentium Quad'
df['Celeron_Dual'] = [1 if x == 'Celeron Dual' else 0 for x in df['processor_name']]
df['M1'] = [1 if x=='M1' else 0 for x in df['processor_name']]
df['Core'] = [3 if x=='Core i3' else (5 if x=='Core i5' else (7 if x=='Core i7' else (9 if x=='Core i9' else 0))) for x in df['processor_name']]
df['Ryzen'] = [3 if x=='Ryzen 3' else (5 if x=='Ryzen 5' else (7 if x=='Ryzen 7' else (9 if x=='Ryzen 9' else 0))) for x in df['processor_name']]
df.drop('processor_name', axis=1, inplace=True)
#to prevent co linearity, we drop one of the new columns from processor name
df.drop('M1', axis=1, inplace=True)




"""Processor_brand
we have considered using the Number of Ratings || Number of Reviews as these could reflect the brand’s popularity and customer engagement.
we have considered creative methods to encode the few coming columns, 
lets start with the processor_brand column, we grouped it according to the number of reviews and printed out the average, 
then we used this average as a weight to encode each brand.
"""
# Calculate the total number of reviews for each processor brand
total_reviews = df.groupby('processor_brand')['Number of Reviews'].sum()
# Normalize the total reviews to a 0-10 scale
max_reviews = total_reviews.max()
encoded_values = (total_reviews / max_reviews) * 10
print("processor brand encoded weights:", encoded_values)

# Define a function to perform target encoding for processor_name
def encode_processor_name(name):
    # Example mapping based on hypothetical target variable (e.g., performance score)
    encoding = {
        'Intel': 10,
        'M1': 1,
        'AMD': 4,
    }
    return encoding.get(name, 0)  # Default to 0 if processor name not found
# Apply the target encoding function to the processor_name column
df['processor_brand'] = df['processor_brand'].apply(encode_processor_name)





"""##Brand column
we have grouped this column based on its popularity among users, 
the popularity among users can be known through the number of ratings, if its high, 
then more users have bought and tried the certain brand of the electronic device
"""
# Calculate the total number of reviews for each processor brand
total_reviews = df.groupby('brand')['Number of Ratings'].sum()
# Normalize the total reviews to a 0-10 scale
max_reviews = total_reviews.max()
encoded_values_Brand = (total_reviews / max_reviews) * 10
print("brand encoded weights:", encoded_values_Brand)

# Define a function to perform target encoding for processor_name
def encode_processor_name(name):
    # Example mapping based on hypothetical target variable (e.g., performance score)
    encoding = {
        'ASUS' : 10,
        'Lenovo' : 2,
        'HP' : 6,
        'APPLE' : 3,
        'DELL' : 2,
        'acer' : 1,
        'MSI' : 1,
        'Avita': 0
    }
    return encoding.get(name, 0)  # Default to 0 if processor name not found
# Apply the target encoding function to the processor_name column
df['brand'] = df['brand'].apply(encode_processor_name)



"""##Label Encoding"""
from sklearn.preprocessing import LabelEncoder
# Create a LabelEncoder object
le = LabelEncoder()
# Loop through each column and label encode categorical columns
for i in df.columns: #iterate over all columns
    if (df[i].dtype == "O"):  # if the column is categorical
        le.fit(df[i])  # fit on the training data
        df[i] = le.transform(df[i])  # transform the training data
        with open(f'{i}_encoder.pkl', 'wb') as file:
            pickle.dump(le, file)


'''now lets fill the nulls of the generation column'''
#to prevent data leakage, we should split train and test before normalizing
X=df.drop(columns=['rating'])
Y=df['rating']


from sklearn.ensemble import RandomForestClassifier
# Function to impute missing values using RandomForestClassifier
def impute_missing_values(X_train, column):
    # Separate the data with and without missing values in the training set
    train_data = X_train[X_train[column].notnull()]
    predict_data_train = X_train[X_train[column].isnull()]
    # Prepare the features and target
    X_train_no_nulls = train_data.drop(column, axis=1)
    y_train_no_nulls = train_data[column]
    # Initialize the classifier
    classifier = RandomForestClassifier(random_state=0)
    # Train the classifier
    classifier.fit(X_train_no_nulls, y_train_no_nulls)
    pickle.dump(classifier, open('GnrtnNullsFill.pkl', 'wb'))  # this serializes the object
    # Predict the missing values in the training set
    X_predict_train = predict_data_train.drop(column, axis=1)
    predicted_values_train = classifier.predict(X_predict_train)
    # Reintegrate the predicted values into the training set
    X_train.loc[X_train[column].isnull(), column] = predicted_values_train
    # Predict the missing values in the testing set
    # Reintegrate the predicted values into the testing set
    return X_train

# Apply the function to the 'processor_gnrtn' column in both the training and testing sets
X_train_initial = impute_missing_values(X, 'processor_gnrtn')






# THIS IS FOR THE DEPLOYMENTTTTTTTTTTTTT
from sklearn import preprocessing
mymy = X_train_initial[['processor_brand', 'processor_gnrtn', 'graphic_card_gb', 'weight',
       'warranty', 'msoffice', 'Pentium_Quad', 'Celeron_Dual', 'Core', 'Ryzen']]
scalerdeploy = preprocessing.MinMaxScaler()
# Select the features to scale
col_names= mymy.columns
# Fit the scaler on the training data (ONLY on X_train)
scalerdeploy.fit(mymy[col_names])
# Transform the training and test data using the fitted scaler
mymy[col_names] = scalerdeploy.transform(mymy[col_names])

pickle.dump(scalerdeploy, open('deploymentScaler.pkl','wb'))




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = (
    train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=0))



'''  TRIAL FOR THE RECURSIVE FEATURE ELIMINATION METHOD '''

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# Initialize the model to be used for feature selection
model = LinearRegression()
# Initialize RFE with the model and the desired number of features
selector = RFE(model, n_features_to_select=10)  # You can change the number of features
# Fit RFE
selector = selector.fit(X_train, y_train)
# Get the mask of the selected features
selected_features_mask = selector.support_
# Apply the mask to get the selected feature names
selected_features = X_train.columns[selected_features_mask]
# Print the selected features and their rankings
print(f'Selected features: {selected_features}')






#X= df.drop(columns=['rating'])
X_train = X_train[selected_features]
X_test = X_test[selected_features]




"""Train using a linear regression"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Create and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
# Make predictions on both the training and testing sets
y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)
# Calculate and print the MSE for both the training and testing sets
mse_train_linear = mean_squared_error(y_train, y_train_pred)
mse_test_linear = mean_squared_error(y_test, y_test_pred)
print(f'Linear Regression - Train MSE: {mse_train_linear}')
print(f'Linear Regression - Test MSE: {mse_test_linear}')
# Calculate and print the R^2 for both the training and testing sets
r2_train_linear = r2_score(y_train, y_train_pred)
r2_test_linear = r2_score(y_test, y_test_pred)
print(f'Linear Regression - Train R^2: {r2_train_linear}')
print(f'Linear Regression - Test R^2: {r2_test_linear}')
pickle.dump(linear_model, open('linear_model.pkl','wb')) #this serializes the object




from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)
mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
mse_test_rf = mean_squared_error(y_test, y_test_pred_rf)
r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
print(f'Random Forest - Train MSE: {mse_train_rf}')
print(f'Random Forest - Test MSE: {mse_test_rf}')
print(f'Random Forest - Train R^2: {r2_train_rf}')
print(f'Random Forest - Test R^2: {r2_test_rf}')


pickle.dump(rf_model, open('rf_model.pkl','wb')) #this serializes the object




