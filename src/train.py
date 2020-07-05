import sys
import os
import yaml
from sklearn.naive_bayes import MultinomialNB
import pickle

# read the command line params
if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py features-dir-path model-filename\n'
    )
    sys.exit(1)

features_path = sys.argv[1]
model_filename = sys.argv[2]

# read pipeline params
params = yaml.safe_load(open('params.yaml'))['train']

alpha = params['alpha']

# load the train features
features_train_pkl = os.path.join(features_path, 'train.pkl')
with open(features_train_pkl, 'rb') as f:
    train_data = pickle.load(f)

X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

# train the model
clf = MultinomialNB(alpha=alpha)
clf.fit(X, y)

# save the model
with open(model_filename, 'wb') as f:
    pickle.dump(clf, f)
