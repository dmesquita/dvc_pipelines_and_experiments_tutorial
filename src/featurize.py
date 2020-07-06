import sys
import os
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# read command line params
if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython featurize.py data-dir-path features-dir-path\n'
    )
    sys.exit(1)

data_path = sys.argv[1]
features_path = sys.argv[2]

os.makedirs(features_path, exist_ok=True)

train_input_file = os.path.join(data_path, 'train.csv')
test_input_file  = os.path.join(data_path, 'test.csv')

# read the data from file
df_train = pd.read_csv(train_input_file)
df_test = pd.read_csv(test_input_file)

def extract_column(column, df_path):
    df = get_df(df_path)
    corpus = df[[column]]

    return corpus

def get_train_and_test_corpus(df_1, df_2):
    corpus_train = df_1["text"]
    corpus_test = df_2["text"]

    return corpus_train.append(corpus_test) 

def append_labels_and_save_pkl(df, tfidf_matrix, filename):
    output_file = os.path.join(features_path, filename)
    target = df[["target"]]
    output = pd.concat([pd.DataFrame(tfidf_matrix.toarray()), target], axis=1)

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)


vectorizer = TfidfVectorizer()
# we need to fit the vectorizer with both train and test data
corpus = get_train_and_test_corpus(df_train, df_test)
vectorizer.fit(corpus)

# transform the data
train_matrix = vectorizer.transform(df_train["text"])
test_matrix = vectorizer.transform(df_test["text"])

# save data to pickle (appending the labels column)
append_labels_and_save_pkl(df_train, train_matrix, 'train.pkl')
append_labels_and_save_pkl(df_test, test_matrix, 'test.pkl')
