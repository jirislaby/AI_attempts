#!/usr/bin/python3

import pandas as pd
import numpy as np
from pickle import dump,load

print("read")
data = pd.read_csv("data2.csv")
#print(data)

from sklearn.utils import class_weight
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
files = np.array(data['file_id']).reshape(-1, 1)
emails = np.array(data['email_id'])
print(files)
print(emails)

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(emails), y=emails)

# Create a mapping of email_id to their respective weight
class_weights_map = {email_id: weight for email_id, weight in zip(np.unique(emails), class_weights)}

# Prepare weights for each sample based on its email_id
sample_weights = np.array([class_weights_map[email_id] for email_id in emails])

#percent = data['percent']
authors = model.fit(files, emails, sample_weight=sample_weights) #percent)

#print(authors)
#print(files)

file = 'fs/btrfs/dev-replace.c'
print(data[data.file == file])
new = data[data.file == file].head(1)['file_id']
new = np.array(new).reshape(-1, 1)
print("looking for: ", new)
pred = model.predict(new)
print(pred, pred[0])
print(data[data.email_id == pred[0]].head(1).email)
