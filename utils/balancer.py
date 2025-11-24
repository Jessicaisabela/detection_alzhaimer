import pandas as pd 
from imblearn.over_sampling import SMOTE 

input_train = pd.read_csv('./archive/train/input_train.csv')
output_train = pd.read_csv('./archive/train/output_train.csv')

original_columns = input_train.columns
input_train_numeric = pd.get_dummies(input_train, drop_first=True)

sm = SMOTE(random_state=42)
input_train_balanced, output_train_balanced = sm.fit_resample(input_train_numeric, output_train)

print(output_train_balanced.value_counts())

input_train_balanced.to_csv('./archive/train/input_train_balanced.csv', index=False)
output_train_balanced.to_csv('./archive/train/output_train_balanced.csv', index=False)
