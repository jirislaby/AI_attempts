#!/usr/bin/python3

import pandas as pd
import numpy as np
from pickle import dump,load
#from sklearn.decomposition import NMF

print("read")
data = pd.read_csv("data2.csv")
#print(data)

from sklearn.naive_bayes import GaussianNB

# Assuming data is already loaded
model = GaussianNB()

# Reshape 'files' to 2D (since it's a single feature)
files = np.array(data['file_id']).reshape(-1, 1)
emails = np.array(data['email_id'])
percent = data['percent']

# Train the model with sample weights
model.fit(files, emails, sample_weight=percent)

# Debugging - Check new file and input
file = 'fs/btrfs/dev-replace.c'
file_data = data[data.file == file]

if file_data.empty:
    print(f"File '{file}' not found in dataset.")
else:
    # Extract the file ID for prediction
    new = file_data.head(1)['file_id']
    new = np.array(new).reshape(1, -1)  # Reshape for prediction

    print("Looking for: ", new)

    # Make a prediction
    pred = model.predict(new)
    print(f"Prediction result: {pred}, first predicted label: {pred[0]}")

    # Find the predicted author
    predicted_author = data[data.email_id == pred[0]]

    if not predicted_author.empty:
        print("Predicted developer email: ", predicted_author.head(1).email)
    else:
        print(f"No developer found for predicted email_id: {pred[0]}")

