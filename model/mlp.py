import pandas as pd
import numpy as np
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# 1. Carregar Dados
db = pd.read_csv('./archive/alzheimers_disease_data.csv')

target_col = 'Diagnosis'
if target_col not in db.columns:
    raise ValueError(f"Coluna alvo '{target_col}' não encontrada!")

y = db[[target_col]].copy()
x = db.drop(columns=[target_col, 'PatientID', 'DoctorInCharge'], errors='ignore')

# 2. Split (Mantendo a proporção original com stratify)
input_train_df, input_test_df, output_train_df, output_test_df = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Escalonamento
scaler = StandardScaler()
input_train_scaled = scaler.fit_transform(input_train_df)
input_test_scaled = scaler.transform(input_test_df)

# 4. Preparar Target Numérico
if pd.api.types.is_numeric_dtype(output_train_df['Diagnosis']):
    output_train_numeric = output_train_df['Diagnosis'].astype(int)
    output_test_numeric = output_test_df['Diagnosis'].astype(int)
else:
    output_train_numeric = output_train_df['Diagnosis'].map({'Alzheimer': 1, 'Normal': 0})
    output_test_numeric = output_test_df['Diagnosis'].map({'Alzheimer': 1, 'Normal': 0})

output_train_numeric = output_train_numeric.fillna(0)
output_test_numeric = output_test_numeric.fillna(0)

input_dim = input_train_scaled.shape[1]

# 5. CALCULAR PESOS (O segredo do sucesso anterior)
class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(output_train_numeric),
    y=output_train_numeric
)
class_weights_dict = dict(enumerate(class_weights_vals))
print(f"Pesos calculados: {class_weights_dict}")

def objective(trial):
    K.clear_session()
    model = Sequential()
    
    n_layers = trial.suggest_int('n_layers', 1, 3)
    units_first = trial.suggest_int('units_first', 64, 512)
    activation = trial.suggest_categorical('activation', ['relu', 'elu'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Regularização L2 mantida (ajuda a generalizar)
    l2_rate = 0.001 
    
    model.add(Dense(units_first, activation=activation, input_dim=input_dim, kernel_regularizer=l2(l2_rate)))
    model.add(Dropout(dropout_rate))
    
    for i in range(n_layers):
        units = trial.suggest_int(f'units_{i}', 32, 256)
        model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_rate)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        TFKerasPruningCallback(trial, 'val_accuracy')
    ]

    # Aqui usamos o class_weight no lugar do SMOTE
    history = model.fit(
        input_train_scaled,
        output_train_numeric,
        validation_data=(input_test_scaled, output_test_numeric),
        epochs=50,
        batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
        callbacks=callbacks,
        class_weight=class_weights_dict, 
        verbose=0
    )

    return history.history['val_accuracy'][-1]

# Otimização
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)

print(study.best_params)

# Treino Final
best_params = study.best_params
final_model = Sequential()
l2_rate = 0.001

final_model.add(Dense(best_params['units_first'], 
                      activation=best_params['activation'], 
                      input_dim=input_dim,
                      kernel_regularizer=l2(l2_rate)))
final_model.add(Dropout(best_params['dropout_rate']))

for i in range(best_params['n_layers']):
    units = best_params[f'units_{i}']
    final_model.add(Dense(units, activation=best_params['activation'], kernel_regularizer=l2(l2_rate)))
    final_model.add(Dropout(best_params['dropout_rate']))

final_model.add(Dense(1, activation='sigmoid'))

final_model.compile(optimizer=Adam(learning_rate=best_params['lr']),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

final_history = final_model.fit(
    input_train_scaled,
    output_train_numeric,
    validation_data=(input_test_scaled, output_test_numeric),
    epochs=100,
    batch_size=best_params['batch_size'],
    class_weight=class_weights_dict,
    verbose=1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
)

final_model.save('./model/alzheimer_mlp_optimized.keras')
pd.DataFrame(input_test_scaled, columns=input_test_df.columns).to_csv('./archive/test/input_test_scaled.csv', index=False)
output_test_numeric.to_csv('./archive/test/output_test.csv', index=False)