import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("Carregando dataset...")
db = pd.read_csv('/home/bruno/detection_alzhaimer/archive/alzheimers_disease_data.csv')

print("\nColunas do dataset:")
print(db.columns)

target_col = 'Diagnosis'

if target_col not in db.columns:
    raise ValueError(f"Coluna alvo '{target_col}' não encontrada no dataset!")

y = db[[target_col]].copy()
x = db.drop(columns=[target_col, 'PatientID', 'DoctorInCharge'], errors='ignore')

print("\nVerificando valores nulos no dataset:")
print(db.isna().sum())

input_train_df, input_test_df, output_train_df, output_test_df = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
input_train_scaled = scaler.fit_transform(input_train_df)
input_test_scaled = scaler.transform(input_test_df)

input_train_scaled = pd.DataFrame(input_train_scaled, columns=input_train_df.columns)
input_test_scaled = pd.DataFrame(input_test_scaled, columns=input_test_df.columns)

print("\n===== Verificando NaNs no input_train_df =====")
print(input_train_df.isna().sum())

print("\n===== Verificando NaNs no input_test_df =====")
print(input_test_df.isna().sum())

print("\n===== Valores únicos na coluna Diagnosis =====")
print(output_train_df['Diagnosis'].unique())

if pd.api.types.is_numeric_dtype(output_train_df['Diagnosis']):
    output_train_numeric = output_train_df['Diagnosis'].astype(int)
    output_test_numeric = output_test_df['Diagnosis'].astype(int)
else:
    output_train_numeric = output_train_df['Diagnosis'].map({'Alzheimer': 1, 'Normal': 0})
    output_test_numeric = output_test_df['Diagnosis'].map({'Alzheimer': 1, 'Normal': 0})

print("\n===== Valores únicos em output_train_numeric (após conversão) =====")
print(pd.Series(output_train_numeric).unique())

print("\n===== Checando valores inválidos =====")
print("NaN em input_train_scaled:", np.isnan(input_train_scaled.values).any())
print("NaN em output_train_numeric:", output_train_numeric.isna().any())
print("Valores únicos em output_train_numeric:", output_train_numeric.unique())
print("Entradas finitas?", np.isfinite(input_train_scaled.values).all())

output_train_numeric = output_train_numeric.fillna(0)
output_test_numeric = output_test_numeric.fillna(0)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,        
    restore_best_weights=True
)

input_dim = input_train_scaled.shape[1]
model = Sequential([
    Dense(512, activation='relu', input_dim=input_dim),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid', name='output_layer') 
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    input_train_scaled,
    output_train_numeric,
    validation_data=(input_test_scaled, output_test_numeric),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)

model.save('/home/bruno/detection_alzhaimer/model/alzheimer_mlp.keras')
input_train_scaled.to_csv('/home/bruno/detection_alzhaimer/archive/train/input_train_scaled.csv', index=False)
input_test_scaled.to_csv('/home/bruno/detection_alzhaimer/archive/test/input_test_scaled.csv', index=False)

print("\nTreinamento concluído e arquivos salvos com sucesso!")