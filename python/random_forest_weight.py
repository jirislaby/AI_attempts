#!/usr/bin/python3
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# Načtení dat
data = pd.read_csv("data.csv")

# Příprava vstupních a výstupních dat
X = data[['file_id']].values  # Pouze ID souboru pro vstup
y = data['email_id'].values  # Výstupní data (target)
weights = data['count'].values  # Počty změn jako váhy

# Kontrola počtu vzorků pro každou třídu
class_counts = pd.Series(y).value_counts()
print("Counts of each class:\n", class_counts)

# Filtrace tříd, které mají méně než 2 vzorky
valid_classes = class_counts[class_counts >= 2].index
filtered_data = data[data['email_id'].isin(valid_classes)]

# Příprava vstupních a výstupních dat po filtrování
X_filtered = filtered_data[['file_id']].values
y_filtered = filtered_data['email_id'].values
weights_filtered = filtered_data['count'].values

# Rozdělení dat na tréninkovou a testovací sadu
#X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
#    X_filtered, y_filtered, weights_filtered, test_size=0.2, random_state=42, stratify=y_filtered
#)
X_train = X_filtered
y_train = y_filtered
weights_train = weights_filtered

# Trénink modelu Random Forest s automatickými váhami
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train, sample_weight=weights_train)  # Použití počtů změn jako vah

# Predikce na testovacích datech
#y_pred = model.predict(X_test)

# Vyhodnocení výkonu modelu
#print(classification_report(y_test, y_pred, zero_division=1))

# Příklad predikce pro nový soubor
file = sys.argv[1]
new = data[data.file == file].head(1).file_id
new = np.array(new).reshape(-1, 1)
print(new)
new_file_id = np.array(new)  # ID souboru, pro který chcete predikovat
predicted_email_id = model.predict(new_file_id)
print("Predicted email_id:", predicted_email_id)
print(data[data.email_id == predicted_email_id[0]].head(1).email)
print(data[data.file == file])
