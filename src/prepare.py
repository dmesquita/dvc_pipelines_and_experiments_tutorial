import os
import sys
import yaml
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# read the command line params
if len(sys.argv) != 2:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 prepare.py data-dir-path\n'
    )
    sys.exit(1)

# create folder to save file
data_path = sys.argv[1]
os.makedirs(data_path, exist_ok=True)
    
# read the pipeline params
params = yaml.safe_load(open('params.yaml'))['prepare']

categories = params['categories']

#fetch data
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

def newsgroups_to_csv(split_name, newsgroups, data_path):
    df = pd.DataFrame([newsgroups.data, newsgroups.target.tolist()]).T
    df.columns = ['text', 'target']

    df_target_names = pd.DataFrame(newsgroups.target_names)
    df_target_names.columns = ['target_name']

    out = pd.merge(df, df_target_names, left_on='target', right_index=True)
    out.to_csv(os.path.join(data_path, split_name+".csv"))

# save data to file
newsgroups_to_csv("train", newsgroups_train, data_path)
newsgroups_to_csv("test", newsgroups_test, data_path)
