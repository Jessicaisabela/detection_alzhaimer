import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("Carregando dataset...")
df = pd.read_csv('./archive/alzheimers_disease_data.csv')

cols_to_drop = ['PatientID', 'DoctorInCharge', 'Diagnosis']
X = df.drop(columns=cols_to_drop, errors='ignore')
y = df['Diagnosis']

for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

print("Analisando importância das colunas...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Importância das Features para Detecção de Alzheimer')
plt.xlabel('Importância (Poder de Decisão)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\n=== Ranking de Importância ===")
print(feature_importance_df)