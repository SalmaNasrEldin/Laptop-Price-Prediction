#importing the libraries
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt

#reading the csv file into a dataframe
df=pd.read_csv('ElecDeviceRatingPrediction_Milestone2.csv')





print(df.isna().sum())



# Processor Generation Distribution by Brand
df.groupby(['brand', 'processor_gnrtn']).size().unstack().plot(kind='pie', subplots=True, figsize=(30, 60))



#now ill try to print the distinct values in each column to see if there are problems with spaces since most of the columns are strings...
for c in df.columns:
    if df[c].dtype != 'int64':
        print(c, df[c].unique())




df['processor_gnrtn'].value_counts()
"""we have noticed that there are 198 'Not Available' values in the processor generation column, 
we can replace them by null (for now), and use this column as my target variable to predict the nulls in the processor generation 
column by feeding it to a random forest model."""

df['processor_gnrtn'] = df['processor_gnrtn'].replace('Not Available', np.nan)
#checking the number of nulls in each column
df.isna().sum()




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




#to prevent data leakage, we should split train and test before normalizing
X_initial=df.drop(columns=['rating'])
y_initial=df['rating']
from sklearn.model_selection import train_test_split
X_train_initial, X_test_initial, y_train, y_test = (
    train_test_split(X_initial, y_initial, test_size=0.2, shuffle=True, random_state=42))


# Print the mode of the columns 'ram_gb', 'ram_type', 'ssd', 'hdd'
for col in ['ram_gb', 'ram_type', 'ssd', 'hdd']:
    print(f"Mode of {col}: {df[col].mode()[0]}")
#The [0] after mode() is used to get the first mode value in case the column has multiple modes.

# Print the median of the columns 'Price', 'Number of Ratings', 'Number of Reviews'
for col in ['Price', 'Number of Ratings', 'Number of Reviews']:
    print(f"Median of {col}: {df[col].median()}")



"""Processor_brand
we have considered using the Number of Ratings || Number of Reviews as these could reflect the brand’s popularity and customer engagement.
we have considered creative methods to encode the few coming columns, 
lets start with the processor_brand column, we grouped it according to the number of reviews and printed out the average, 
then we used this average as a weight to encode each brand.
"""
# Calculate the total number of reviews for each processor brand
total_reviews = X_train_initial.groupby('processor_brand')['Number of Reviews'].sum()
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
total_reviews = X_train_initial.groupby('brand')['Number of Ratings'].sum()
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





"""##Using one-hot encoder"""
'''from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(drop='first')# drop first is to avoid multicollinearity(It means repetition of data)
warranty_enocded=ohe.fit_transform(df[['warranty']])
warranty_enocded=warranty_enocded.toarray() # converting to numpy array
warranty_enocded=pd.DataFrame(warranty_enocded) # converting to dataframe
warranty_enocded=warranty_enocded.add_prefix('warranty ')
df=pd.concat([df,warranty_enocded],axis=1) # append to original dataframe
df.tail()#you can view new column at rightmost end
'''







"""##Label Encoding"""
from sklearn.preprocessing import LabelEncoder
# Create a LabelEncoder object
le = LabelEncoder()
# Loop through each column and label encode categorical columns
for i in X_train_initial.columns: #iterate over all columns
    if (X_train_initial[i].dtype == "O"):  # if the column is categorical
        le.fit(X_train_initial[i])  # fit on the training data
        X_train_initial[i] = le.transform(X_train_initial[i])  # transform the training data
        X_test_initial[i] = le.transform(X_test_initial[i])  # transform the test data
        with open(f'{i}_encoder.pkl', 'wb') as file:
            pickle.dump(le, file)

labels = LabelEncoder()
labels.fit(y_train)
y_train = labels.transform(y_train)  # transform the training data
y_test = labels.transform(y_test)  # transform the test data
pickle.dump(labels, open('labels.pkl','wb'))



'''now lets fill the nulls of the generation column'''

from sklearn.ensemble import RandomForestClassifier
# Function to impute missing values using RandomForestClassifier
def impute_missing_values(X_train, X_test, column):
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
    predict_data_test = X_test[X_test[column].isnull()]
    X_predict_test = predict_data_test.drop(column, axis=1)
    predicted_values_test = classifier.predict(X_predict_test)
    # Reintegrate the predicted values into the testing set
    X_test.loc[X_test[column].isnull(), column] = predicted_values_test
    return X_train, X_test

# Apply the function to the 'processor_gnrtn' column in both the training and testing sets
X_train_initial, X_test_initial = impute_missing_values(X_train_initial, X_test_initial, 'processor_gnrtn')



from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
# Select the features to scale
col_names= X_train_initial.columns
# Fit the scaler on the training data (ONLY on X_train)
scaler.fit(X_train_initial[col_names])
# Transform the training and test data using the fitted scaler
X_train_initial[col_names] = scaler.transform(X_train_initial[col_names])
X_test_initial[col_names] = scaler.transform(X_test_initial[col_names])

pickle.dump(scaler, open('scaler.pkl','wb'))





'''  ANOVA '''
from sklearn.feature_selection import SelectKBest, f_classif

# Create an instance of SelectKBest with the desired number of features
k = 15  # Number of top features to select
selector = SelectKBest(f_classif, k=k)

# Fit the selector to the training data
selector.fit(X_train_initial, y_train)

# Get the p-values and scores for each feature
p_values = selector.pvalues_
scores = selector.scores_

# Get the indices of the selected features
selected_indices = selector.get_support(indices=True)

# Get the names of the selected features
selected_features = [X_train_initial.columns[idx] for idx in selected_indices]

# Print the selected features
print(selected_features)




#without the drop() it calculated the correlation between target and itself. i don't need that.
#df.drop('rating', axis=1).corrwith(df['rating']).plot(kind = 'bar', grid = True, figsize = (12, 8),
#                                                     title = "Correlation with Target")




#X= df.drop(columns=['rating'])
X_train = X_train_initial[selected_features]
X_test = X_test_initial[selected_features]



"""Train using a logistic regression"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create and train the linear regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Make predictions on both the training and testing sets
y_test_pred = logistic_model.predict(X_test)
y_train_pred = logistic_model.predict(X_train)

accuracy1 = accuracy_score(y_train, y_train_pred)
print("Logistic Regression Train Accuracy:", accuracy1)
accuracy = accuracy_score(y_test, y_test_pred)
print("Logistic Regression Test Accuracy:", accuracy)

pickle.dump(logistic_model, open('logistic_model.pkl','wb')) #this serializes the object




"""train using SVM"""
from sklearn.model_selection import GridSearchCV
from sklearn import svm
###################WE PERFORMED GRID SEARCH TO FID BEST PARAMETER


'''# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
    'kernel': ['rbf', 'linear', 'poly']  # Type of SVM
}

# Initialize the SVM model
svm_model = svm.SVC()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)'''
#Best parameters: {'C': 100, 'gamma': 1, 'kernel': 'poly'}
# Create and train the SVM model
svm_model = svm.SVC(C = 100, gamma = 0.1, kernel = 'rbf')
svm_model.fit(X_train, y_train)

# Make predictions on both the training and testing sets
y_test_pred = svm_model.predict(X_test)
y_train_pred = svm_model.predict(X_train)

accuracy1 = accuracy_score(y_train, y_train_pred)
print("SVM Train Accuracy:", accuracy1)
accuracy = accuracy_score(y_test, y_test_pred)
print("SVM Test Accuracy:", accuracy)

pickle.dump(svm_model, open('svm_model.pkl','wb')) #this serializes the object




"""train using RandomForest"""
######################## WE PERFORMED CROSS VALIDATION IN ORDER TO FIND THE BEST PARAMETERS

'''from sklearn.model_selection import GridSearchCV
# Define the parameter grid
param_grid = {
    'max_depth': [10,15, 20],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10, 15],
    'n_estimators': [100, 200, 250, 300]
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

'''
#Best parameters: {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 15, 'n_estimators': 100}


from sklearn.ensemble import RandomForestClassifier
# Create and train the Random Forest model
#random_forest_model = RandomForestClassifier()
random_forest_model = RandomForestClassifier(max_depth= 10, min_samples_leaf= 4, min_samples_split= 15,
                                             n_estimators= 100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions on both the training and testing sets
y_test_pred = random_forest_model.predict(X_test)
y_train_pred = random_forest_model.predict(X_train)

accuracy1 = accuracy_score(y_train, y_train_pred)
print("RandomForest Train Accuracy:", accuracy1)
accuracy = accuracy_score(y_test, y_test_pred)
print("RandomForest Test Accuracy:", accuracy)

pickle.dump(random_forest_model, open('random_forest_model.pkl','wb'))

