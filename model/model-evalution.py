import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

model = load_model('./model/alzheimer_mlp_optimized.keras')
input_test_scaled = pd.read_csv('./archive/test/input_test_scaled.csv')
y_true = pd.read_csv('./archive/test/output_test.csv')
y_true = y_true.values.flatten().astype(int)

predictions = model.predict(input_test_scaled)
y_pred = (predictions > 0.30).astype(int)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Alzheimer'], 
            yticklabels=['Normal', 'Alzheimer'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

print(classification_report(y_true, y_pred, target_names=['Normal', 'Alzheimer']))