# churn_model


This repo contains some ML exploration on to predict churn using the dataset in `data/Churn_Modelling.csv`. The main file to follow the exploration is this [notebook](notebooks/model_exploration.ipynb)
which relies on some helpers functions in [utils](utils.py), [meta_model](meta_model.py) and [model_comparison](model_comparison.py).

In order to run the notebook, please install the [requirements](requirements.txt), ideally in a new environment. Lightgbm might require you to install libomp, which can be done
with `brew install libomp` on macOS or `sudo apt-get install -y libomp-dev` on Linux. If using conda, `conda install lightgbm==3.2.1` takes care of everything.

The metrics guiding the exploration is the precision at 90% recall. The reasoning behind that choice is that we do not want to miss customers that might churn, so our recall should be pretty high (here I chose 90%, but that could be updated after discussion with the product owner). Therefore, if we want our recall to be at least 90%, the metric to compare is precision (ie out of all the churn predictions, how many actually churned). 
This assumes that the cost when reaching out to a customer predicted to churn is low (since we will be reaching out to a lot of them, even though not all were actually about to churn). It seemed like a reasonable assumption if we are to sent an email with a special offer. 
We will also monitor the area under the curve of the Receiver Operating Characteristic (roc_auc), since it is a correlated metric (higher roc_auc means higher precision at 90% recall) which might be more familiar to ML practioners.

At the end of the exploration, the best performing model is saved to [inference/model/tuned_lgbm.pickle](inference/model/tuned_lgbm.pickle) which is exposed in an API via [opyrator](https://github.com/ml-tooling/opyrator).
This has been packaged into a docker image, which can be pulled with

`docker pull jgleyzes/churn_model:v1`

Then, it can be run with 

`docker run -p 8051:8051 -t -i jgleyzes/churn_model:v1`

The app can be accessed in your browser at `http://0.0.0.0:8051/`
