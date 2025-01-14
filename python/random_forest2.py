#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight

# Load the dataset
data = pd.read_csv("data2.csv")

# One-hot encode the file paths
encoder = OneHotEncoder(sparse_output=False)
file_encoded = encoder.fit_transform(data[['file']])

# Prepare the feature matrix (including count and percent as features)
# Ensure count and percent are treated as 2D arrays
count_percent = data[['count', 'percent']].values  # This should already be 2D
features = np.hstack((file_encoded, count_percent))  # Stack the arrays horizontally

emails = np.array(data['email_id'])

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(emails), y=emails)

# Create a mapping of email_id to their respective weight
class_weights_map = {email_id: weight for email_id, weight in zip(np.unique(emails), class_weights)}

# Prepare weights for each sample based on its email_id
sample_weights = np.array([class_weights_map[email_id] for email_id in emails])

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
model.fit(features, emails)

# Predicting for a new file
file = 'fs/btrfs/dev-replace.c'
new_file_encoded = encoder.transform([[file]])  # One-hot encode the new file
new_features = np.hstack((new_file_encoded, np.array([[0, 0]])))  # Example count and percent

# Make a prediction
pred = model.predict(new_features)
print(f"Prediction result: {pred}, first predicted label: {pred[0]}")

# Find the predicted author
predicted_author = data[data.email_id == pred[0]]
if not predicted_author.empty:
    print("Predicted developer email: ", predicted_author.head(1).email)
else:
    print(f"No developer found for predicted email_id: {pred[0]}")

