import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert to numpy arrays with consistent shape
data = np.array(data_dict['data'], dtype=object)
labels = np.array(data_dict['labels'])

# Verify all samples have 42 features
valid_indices = [i for i,x in enumerate(data) if len(x) == 42]
data = np.array([data[i] for i in valid_indices])
labels = np.array([labels[i] for i in valid_indices])

if len(data) == 0:
    raise ValueError("No valid training samples found (all samples must have 42 features)")

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
