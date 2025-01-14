#!/usr/bin/python3

import pandas as pd
import numpy as np
from pickle import dump,load

print("read")
data = pd.read_csv("data2.csv")
#print(data)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
files = np.array(data['file_id']).reshape(-1, 1)
emails = np.array(data['email_id'])
print(files)
print(emails)

#percent = data['percent']
authors = model.fit(files, emails)

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
