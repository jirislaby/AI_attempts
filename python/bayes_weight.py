#!/usr/bin/python3
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# Načtení dat
data = pd.read_csv("data.csv")

# Příprava vstupních a výstupních dat
X = data[['file_id']].values  # Pouze ID souboru pro vstup
y = data['email_id'].values  # Výstupní data (target)
weights = data['count'].values  # Počty změn jako váhy

# Rozdělení dat na tréninkovou a testovací sadu
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# Trénink modelu s váhami
model = GaussianNB()
model.fit(X_train, y_train, sample_weight=weights_train)  # Použití počtů změn jako vah

# Predikce na testovacích datech
y_pred = model.predict(X_test)

# Vyhodnocení výkonu modelu s zero_division
print(classification_report(y_test, y_pred, zero_division=1))

# Příklad predikce pro nový soubor
new_file_id = np.array([[1911]])  # ID souboru, pro který chcete predikovat
predicted_email_id = model.predict(new_file_id)
print("Predicted email_id:", predicted_email_id)

