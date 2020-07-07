# Building a maintainable Machine Learning pipeline using DVC

This guides uses the [DVC Get Started Guide](https://github.com/iterative/example-get-started)
as a starting point and takes you on **how
to build maintainable Machine Learning pipelines using DVC**.

If you have some time you can check the full article [here](https://towardsdatascience.com/the-ultimate-guide-to-building-maintainable-machiane-learning-pipelines-using-dvc-a976907b2a1b) (it has more in depth explanations than this readme :wink:)

The principles are:
- **Write a python script** for each pipeline step
- **Save the parameters** each script uses in a `yaml` file
- Specify the files each script **depends on**
- Specify the files each script **generates**

In this tutorial we're going to build a model to classify the 20newsgroups dataset.

*Environment*: Linux with **Python 3**, **pip** and **Git** installed

## First: installing DVC as a Python library
```console
$ mkdir dvc_tutorial
$ cd dvc_tutorial
$ python3 -m venv .env
$ source .env/bin/activate
(.env)$ pip3 install dvc
(.env)$ git init
(.env)$ dvc init
```

## 1 - Create a `params.yaml` file
```
# file params.yaml
prepare:
    categories:
        - comp.graphics
        - sci.space
```

## 2 - Create the `prepare.py` script
Save the file  `prepare.py` file (it's available here on this repo) inside `/src`. Your folder structure should look like this:
```
├── params.yaml
└── src
    └── prepare.py
```

## 3 - Create the `prepare.py` stage usinf DVC
The steps for doing that are:
- Write a python script: `prepare.py`
- Save the parameters: `categories` inside `params.yaml`
- Specify the files the script depends on: `prepare.py`
- Specify the files the script generates: the folder `data/prepared`
- Defined the command line instruction to run this step

```console
(.env)$ pip install pyyaml scikit-learn pandas

(.env)$ dvc run -n prepare -p prepare.categories -d src/prepare.py -o data/prepared python3 src/prepare.py
```

## 4 - Create the scripts and the stages for all the other steps
```
(.env)$ dvc run -n featurize -d src/featurize.py -d data/prepared -o data/features python3 src/featurize.py data/prepared data/features

(.env)$ dvc run -n train -p train.alpha -d src/train.py -d data/features -o model.pkl python3 src/train.py data/features model.pkl

(.env)$ dvc run -n evaluate -d src/evaluate.py -d model.pkl -d data/features --metrics-no-cache scores.json --plots-no-cache plots.json python3 src/evaluate.py model.pkl data/features scores.json plots.json
```

## 5 - Change parameters
```
# file params.yaml
prepare:
    categories:
        - comp.graphics
        - rec.sport.baseball
train:
    alpha: 0.9
```
## 6 - Run the pipeline
```console
(.env)$ dvc repro
```

## 7 -  Compare the metrics
```console
(.env)$ dvc params diff

(.env)$ dvc metrics diff
```

## 8 - Visualize and compare metrics using plots
```console
(.env)$ dvc plots show -y precision -x recall plots.json

(.env)$ dvc plots diff --targets plots.json -y precision
```
