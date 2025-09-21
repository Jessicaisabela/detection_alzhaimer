import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
from tensorflow.keras.callbacks import EarlyStopping 
import matplotlib.pyplot as plt 


input_train_df = pd.read_csv(r'C:\Users\jessi\OneDrive\Área de Trabalho\detection_alzhaimer\archive\train\input_train_balanced.csv')
output_train_df = pd.read_csv(r'C:\Users\jessi\OneDrive\Área de Trabalho\detection_alzhaimer\archive\train\output_train_balanced.csv')

input_test_df = pd.read_csv(r'C:\Users\jessi\OneDrive\Área de Trabalho\detection_alzhaimer\archive\test\input_test.csv')
output_test_df = pd.read_csv(r'C:\Users\jessi\OneDrive\Área de Trabalho\detection_alzhaimer\archive\test\output_test.csv')

print("Verificando NaNs no input_train_df:")
print(input_train_df.isnull().sum())
print("\nVerificando NaNs no input_test_df:")
print(input_test_df.isnull().sum())

input_train_numeric = pd.get_dummies(input_train_df, drop_first=True)
input_test_numeric = pd.get_dummies(input_test_df, drop_first=True)

train_cols = input_train_numeric.columns
input_test_aligned = input_test_numeric.reindex(columns=train_cols, fill_value=0)

output_train_numeric = output_train_df['Age'].map({'Yes': 1, 'No': 0})
output_test_numeric = output_test_df['Age'].map({'Yes': 1, 'No': 0})

scaler = StandardScaler() 

input_train_scaled = scaler.fit_transform(input_train_numeric)
input_test_scaled = scaler.transform(input_test_aligned)

output_path = r'C:\Users\jessi\OneDrive\Área de Trabalho\detection_alzhaimer\archive'
pd.DataFrame(input_train_scaled, columns=train_cols).to_csv(f'{output_path}/train/input_train_scaled.csv', index=False)
pd.DataFrame(input_test_scaled, columns=train_cols).to_csv(f'{output_path}/test/input_test_scaled.csv', index=False)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_train_scaled.shape[1],)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
])

model.summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(input_train_scaled, output_train_numeric, epochs=100, batch_size=32, 
                    validation_split=0.2, callbacks=[early_stopping])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(r'C:\Users\jessi\OneDrive\Área de Trabalho\detection_alzhaimer\plot\loss_plot.png')
plt.show()  

pd.DataFrame(history.history).to_csv(r'C:\Users\jessi\OneDrive\Área de Trabalho\detection_alzhaimer\plot\history.csv', index=False)

model.save(r'C:\Users\jessi\OneDrive\Área de Trabalho\detection_alzhaimer\plot\model.h5')