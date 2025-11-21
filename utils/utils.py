import pandas as pd
from sklearn.model_selection import train_test_split 

db = pd.read_csv('./archive/alzheimers_disease_data.csv')

print(db.head())
y = db['Diagnosis']
x = db.drop(['Diagnosis', 'PatientID', 'DoctorInCharge'], axis='columns')

classes = y.value_counts()

print(classes)

#divisao dos dados
input_train, input_test, output_train, output_test = train_test_split(x, y, stratify=y,test_size=0.2)

input_train = pd.DataFrame(input_train)

input_train.to_csv('./archive/train/input_train.csv', index=False)

input_test = pd.DataFrame(input_test)

input_test.to_csv('./archive/test/input_test.csv', index=False)
output_train = pd.DataFrame (output_train)

output_train.to_csv('./archive/train/output_train.csv', index=False)

output_test = pd.DataFrame(output_test)

output_test.to_csv('./archive/test/output_test.csv', index=False)