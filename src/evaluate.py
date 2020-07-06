import sys
import os
from sklearn.metrics import precision_recall_curve, auc
import pickle
import json

# read command line parameters
if len(sys.argv) != 5:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 evaluate.py model-filename features-dir-path scores-filename\
                plots-filename\n'
    )
    sys.exit(1)

model_filename = sys.argv[1]
features_path = sys.argv[2]
test_features_file = os.path.join(os.path.join(features_path, 'test.pkl'))
scores_file = sys.argv[3]
plots_file = sys.argv[4]

# load features
with open(test_features_file, 'rb') as f:
    test_features = pickle.load(f)
    
X_test = test_features.iloc[:,:-1]
y_test = test_features.iloc[:,-1]

# load model
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

# make predictions
predictions_by_class = model.predict_proba(X_test)
predictions = predictions_by_class[:,-1]

# generate scores
precision, recall, thresholds = precision_recall_curve(y_test, predictions)
auc = auc(recall, precision)

# save scores
with open(scores_file, 'w') as f:
    json.dump({'auc': auc}, f)

# save plots
with open(plots_file, 'w') as f:
    proc_dict = {'proc': [{
        'precision': p,
        'recall': r,
        'threshold': t
        } for p, r, t in zip(precision, recall, thresholds)
    ]}
    json.dump(proc_dict, f)
