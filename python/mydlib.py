#!/usr/bin/python3

import pandas as pd
import numpy as np
import dlib

# Load the dataset
data = pd.read_csv("data2.csv")

# Prepare the feature matrix (file_id, count, and percent as features)
features = data[['file_id', 'count', 'percent']].values  # Use file_id, count, and percent as features

# Prepare labels (email_id as integer)
emails = np.array(data['email_id'])  # Assume email_id is already an integer

# Convert features and labels to dlib's matrix format
features_dlib = dlib.matrix(features.tolist())  # Convert to list of lists for dlib
labels_dlib = dlib.vector(emails.tolist())      # Convert labels to dlib's vector

# Initialize the SVM trainer
trainer = dlib.svm_multiclass_linear_trainer()

# Train the SVM model
model = trainer.train(features_dlib, labels_dlib)

# Predicting for a new file
file_id = 1911  # Example file_id for 'fs/btrfs/dev-replace.c', use actual file_id from your data
new_count = 0   # Example count, modify as needed
new_percent = 0.0  # Example percent, modify as needed

# Prepare the new features as a dlib matrix
new_features = np.array([[file_id, new_count, new_percent]])
new_features_dlib = dlib.matrix(new_features.tolist())

# Make a prediction
pred = model(new_features_dlib)
predicted_email_id = labels_dlib[pred]  # Get the predicted email_id

# Find the predicted author
predicted_author = data[data.email_id == predicted_email_id]
if not predicted_author.empty:
    print("Predicted developer email: ", predicted_author.head(1).email.values[0])
else:
    print(f"No developer found for predicted email_id: {predicted_email_id}")


