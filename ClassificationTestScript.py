#importing the libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score


#reading the csv file into a dataframe
df=pd.read_csv('ElecClassTest.csv')

# Fill nulls in 'ram_gb', 'ram_type', 'ssd', 'hdd' with the mode of each respective column
df['ram_gb'].fillna('8 GB', inplace=True)
df['ram_type'].fillna('DDR4', inplace=True)
df['ssd'].fillna('512 GB', inplace=True)
df['hdd'].fillna('0 GB', inplace=True)

# Fill nulls in 'os' with '64-bit Mac' if the 'brand' == APPLE, else fill it with '32-bit Windows'
# Assuming 'brand' is a column in your DataFrame
df.loc[(df['os'].isnull()) & (df['brand'] == 'APPLE'), 'os'] = '64-bit Mac'
df['os'].fillna('32-bit Windows', inplace=True)

# Fill nulls in 'weight' with 'Casual'
df['weight'].fillna('Casual', inplace=True)

# Fill nulls in 'warranty' with 'No warranty'
df['warranty'].fillna('No warranty', inplace=True)

# Fill nulls in 'Touchscreen' and 'msoffice' with 'No'
for col in ['Touchscreen', 'msoffice']:
    df[col].fillna('No', inplace=True)

# Fill nulls in 'Price', 'Number of Ratings' and 'Number of Reviews' with the median of each respective column
df['Price'].fillna(64990, inplace=True)
df['Number of Ratings'].fillna(17, inplace=True)
df['Number of Reviews'].fillna(2, inplace=True)


#there are 'Not Available' values in 'gnrtn' column, we'll remove it and use it as my target to predict said values
df['processor_gnrtn'] = df['processor_gnrtn'].replace('Not Available', np.nan)


def convert_to_float(gnrtn):
    if pd.isnull(gnrtn):
        return gnrtn
    else:
        return int(gnrtn.rstrip('th'))
# Apply the function to the 'processor_gnrtn' column
df['processor_gnrtn'] = df['processor_gnrtn'].apply(convert_to_float)


"""##Changing all features that have string extension to number only"""
df = df.replace('GB', '', regex=True)


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


#i will load only the features that will be used by my model
warranty = pickle.load(open('warranty_encoder.pkl', 'rb'))
msoffice = pickle.load(open('msoffice_encoder.pkl', 'rb'))
os = pickle.load(open('os_encoder.pkl', 'rb'))
ram_type = pickle.load(open('ram_type_encoder.pkl', 'rb'))
touchscreen = pickle.load(open('Touchscreen_encoder.pkl', 'rb'))
weight = pickle.load(open('weight_encoder.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))

df['warranty'] = warranty.transform(df['warranty'])
df['msoffice'] = msoffice.transform(df['msoffice'])
df['os'] = os.transform(df['os'])
df['ram_type'] = ram_type.transform(df['ram_type'])
df['Touchscreen'] = touchscreen.transform(df['Touchscreen'])
df['weight'] = weight.transform(df['weight'])
df['rating'] = labels.transform(df['rating'])


#filling the nulls(if any) of the 'processor_gnrtn' column
classifier = pickle.load(open('GnrtnNullsFill.pkl', 'rb'))
# Predict the missing values in the training set
predict_data_train = df[df['processor_gnrtn'].isnull()]
X_predict_train = predict_data_train.drop(['processor_gnrtn','rating'], axis=1)
predicted_values_train = classifier.predict(X_predict_train)
# Reintegrate the predicted values into the training set
df.loc[df['processor_gnrtn'].isnull(), 'processor_gnrtn'] = predicted_values_train

y = df['rating']
df = df.drop(columns=['rating'])

scaler = pickle.load(open('scaler.pkl', 'rb'))
'''df = scaler.transform(df)

df = df[['processor_brand', 'ram_gb', 'ram_type', 'hdd', 'os', 'weight', 'warranty', 'Touchscreen', 'msoffice', '
Price', 'Number of Ratings', 'Number of Reviews', 'Pentium_Quad', 'Celeron_Dual', 'Core']]
'''

# Assuming 'scaler' returns a NumPy array, convert it back to a DataFrame
df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

df = df_scaled[['processor_brand', 'ram_gb', 'ram_type', 'hdd', 'os', 'weight', 'warranty', 'Touchscreen', 'msoffice',
                'Price', 'Number of Ratings', 'Number of Reviews', 'Pentium_Quad', 'Celeron_Dual', 'Core']]



# Make a prediction
random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))
random_forest_model = random_forest_model.predict(df)
print(random_forest_model)
train_accuracy = accuracy_score(y, random_forest_model)
print("Random Forest Train accuracy:", train_accuracy)


svm_model = pickle.load(open('svm_model.pkl', 'rb'))
svm_model = svm_model.predict(df)
print(svm_model)
train_accuracy = accuracy_score(y, svm_model)
print("SVM Train accuracy:", train_accuracy)





