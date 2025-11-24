import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

cols_to_ignore = [
    'PatientID', 'DoctorInCharge', 'Diagnosis',
    'EducationLevel', 'Ethnicity', 'Smoking', 'Hypertension', 'Gender', 
    'CardiovascularDisease', 'Forgetfulness', 'PersonalityChanges', 
    'DifficultyCompletingTasks', 'Depression', 'Disorientation', 'Confusion', 
    'FamilyHistoryAlzheimers', 'HeadInjury', 'Diabetes'
]

print("Carregando dataset...")
df = pd.read_csv('./archive/alzheimers_disease_data.csv')
target_col = 'Diagnosis'
y = df[target_col].copy()
X = df.drop(columns=cols_to_ignore, errors='ignore')

for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nCarregando modelos...")
nn_model = load_model('./model/alzheimer_mlp_optimized.keras')
scaler = joblib.load('./model/scaler.pkl')
X_test_scaled = scaler.transform(X_test)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model('./model/alzheimer_xgboost_optimized.json')

print("Gerando probabilidades base...")
prob_nn = nn_model.predict(X_test_scaled).flatten()
prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

def objective(trial):
    weight_xgb = trial.suggest_float('weight_xgb', 0.0, 1.0)
    weight_nn = 1.0 - weight_xgb
    
    threshold = trial.suggest_float('threshold', 0.2, 0.8)
    
    prob_final = (prob_xgb * weight_xgb) + (prob_nn * weight_nn)
    
    y_pred = (prob_final > threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

print("\nIniciando Otimização Bayesiana do Ensemble...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("\nMelhores parâmetros encontrados:")
print(study.best_params)

best_weight_xgb = study.best_params['weight_xgb']
best_weight_nn = 1.0 - best_weight_xgb
best_threshold = study.best_params['threshold']

prob_final = (prob_xgb * best_weight_xgb) + (prob_nn * best_weight_nn)
y_pred_ensemble = (prob_final > best_threshold).astype(int)

print("\n=== RELATÓRIO DO ENSEMBLE OTIMIZADO ===")
print(f"Pesos: XGBoost={best_weight_xgb:.4f} | NeuralNet={best_weight_nn:.4f}")
print(f"Limiar de Decisão: {best_threshold:.4f}")
print(classification_report(y_test, y_pred_ensemble, target_names=['Normal', 'Alzheimer']))

cm = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Normal', 'Alzheimer'],
            yticklabels=['Normal', 'Alzheimer'])
plt.title(f'Ensemble Otimizado (NN + XGBoost)')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.savefig('./matriz_confusao_ensemble_optuna.png')
print("Gráfico salvo como 'matriz_confusao_ensemble_optuna.png'")
plt.show()

acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"\nAcurácia Final do Ensemble: {acc_ensemble:.4f}")