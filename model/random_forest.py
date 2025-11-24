import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Configuração e Carregamento
cols_to_ignore = [
    'PatientID', 'DoctorInCharge', 'Diagnosis',
    'EducationLevel', 'Ethnicity', 'Smoking', 'Hypertension', 'Gender', 
    'CardiovascularDisease', 'Forgetfulness', 'PersonalityChanges', 
    'DifficultyCompletingTasks', 'Depression', 'Disorientation', 'Confusion', 
    'FamilyHistoryAlzheimers', 'HeadInjury', 'Diabetes'
]

print("Carregando dataset...")
df = pd.read_csv('./archive/alzheimers_disease_data.csv')
y = df['Diagnosis'].copy()
X = df.drop(columns=cols_to_ignore, errors='ignore')

# Encoder
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDataset pronto: {X_train.shape[0]} amostras de treino.")
print("-" * 50)

# ==========================================
# DESAFIANTE 1: RANDOM FOREST
# ==========================================
print("1. Treinando RANDOM FOREST...")
rf_model = RandomForestClassifier(
    n_estimators=200, 
    class_weight='balanced', # Crucial para manter o Recall alto
    random_state=42,
    n_jobs=-1 # Usa todos os núcleos do processador (Velocidade máxima)
)

start_time = time.time()
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"Tempo de Treino (RF): {rf_time:.4f} segundos")
print(f"Acurácia (RF): {rf_acc:.4f}")

# ==========================================
# DESAFIANTE 2: XGBOOST (Otimizado)
# ==========================================
print("\n2. Treinando XGBOOST...")
# Usando parâmetros próximos do seu otimizado
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=2.0,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start_time

xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

print(f"Tempo de Treino (XGB): {xgb_time:.4f} segundos")
print(f"Acurácia (XGB): {xgb_acc:.4f}")

# ==========================================
# VEREDITO
# ==========================================
print("-" * 50)
print("RELATÓRIO DETALHADO DO RANDOM FOREST:")
print(classification_report(y_test, rf_pred, target_names=['Normal', 'Alzheimer']))

print("RELATÓRIO DETALHADO DO XGBOOST:")
print(classification_report(y_test, xgb_pred, target_names=['Normal', 'Alzheimer']))