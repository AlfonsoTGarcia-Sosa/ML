import sys
import tensorflow as tf
print("tf.__version__: %s" % str(tf.__version__))
import os
import deepchem as dc
print("dc.__version__: %s" % str(dc.__version__))
from deepchem.utils.save import load_from_disk
import pandas as pd
print(pd)
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
"import seaborn as sns"
data = pd.DataFrame()
pd.__version__
"""Download training set"""
current_dir = os.path.dirname(os.path.realpath("__file__"))
dc.utils.download_url("https://hermes.chem.ut.ee/~alfx/ML/train.csv",current_dir)
df = pd.read_csv("train.csv")
df_rows, df_cols = df.shape
df
PandasTools.AddMoleculeColumnToFrame(df, "Canonical_QSARr","Mol")
"""Download training set binders"""
current_dir = os.path.dirname(os.path.realpath("__file__"))
dc.utils.download_url("https://hermes.chem.ut.ee/~alfx/ML/binders.csv",  current_dir)
active_df = pd.read_csv("binders.csv")
active_df_rows, active_df_cols = active_df.shape
active_df
PandasTools.AddMoleculeColumnToFrame(active_df,"SMILES","Mol")
print([str(x) for x in active_df.columns])
"""Download training set nonbinders"""
current_dir = os.path.dirname(os.path.realpath("__file__"))
dc.utils.download_url("https://hermes.chem.ut.ee/~alfx/ML/nonbinders.csv",  current_dir)
inactive_df = pd.read_csv("nonbinders.csv")
tmp_df = active_df.append(inactive_df)
"""Add label to binders"""
active_df["label"] = ["Binders"]*active_df_rows
PandasTools.AddMoleculeColumnToFrame(inactive_df,"SMILES","Mol")
"""Add label to nonbinders"""
inactive_df_rows, inactive_df_cols = inactive_df.shape
inactive_df["label"] = ["Nonbinders"]*inactive_df_rows
"""Join binders and nonbinders into one dataset"""
tmp_df = active_df.append(inactive_df)
"""Balance dataset by numbers of binders/nonbinders"""
bal_df = tmp_df
g = bal_df.groupby('bindingClass')
g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
g
g.head()
g.shape
user_specified_features = ['HumDockScore','RatDockScore','ChimpDockScore','AVG','P_Act_dockChimp','P_Inact','PredBayes','ratio','avgD_Act','avgD_Inact','PredBindingClass','predMLogR']
from itertools import islice
"from IPython.display import Image, display, HTML"
import tempfile, shutil
"from iPython.display import Image, display, HTML"
"""Featurizer using own features (user_specified_features)"""
featurizer = dc.feat.UserDefinedFeaturizer(user_specified_features)
loader = dc.data.UserCSVLoader(
      tasks=["bindingClass"], id_field="casrn",
      featurizer=featurizer)
g.to_csv('balanced.csv',index=False)
g.columns
g.drop(columns=['s_sd_Canonical\_QSARr','s_m_title','cas','cid', 'gsid','dsstox_substance_id', 'preferred_name','InChI_Code_QSARr','InChI Key_QSARr', 'BindingClass'])
g.to_csv('balanced_droppedcols.csv',index=False)
"""Re-upload training set, now balanced, and with less properties"""
current_dir = os.path.dirname(os.path.realpath("__file__"))
dc.utils.download_url("https://hermes.chem.ut.ee/~alfx/ML/balanced_droppedcols1.csv",  current_dir)
dataset_file = "balanced_droppedcols1.csv"
dataset = loader.featurize(dataset_file)
dataset
"""Download evaluation set"""
current_dir = os.path.dirname(os.path.realpath("__file__"))
dc.utils.download_url("https://hermes.chem.ut.ee/~alfx/ML/Eval_4.csv",  current_dir)
eval_df = pd.read_csv("Eval_4.csv")
eval_df
bal_eval = eval_df
eval_dataset_file = "Eval_4.csv"
eval_dataset = loader.featurize(eval_dataset_file)
valid_dataset = eval_dataset
tmp_df
tmp_df.to_csv('tmp_df.csv', index=False)
tmp_dataset_file = "tmp_df.csv"
tmp_dataset = loader.featurize(tmp_dataset_file)

transformers = [
    dc.trans.NormalizationTransformer(transform_X=True, dataset=dataset),
    dc.trans.ClippingTransformer(transform_X=True, dataset=dataset)]
#datasets = [dataset]
#datasets = [dataset,eval_df]
datasets = [dataset, valid_dataset, tmp_dataset]
for i, dataset in enumerate(datasets):
  for transformer in transformers:
      datasets[i] = transformer.transform(dataset)

train_dataset, valid_dataset, tmp_dataset = datasets
"""Fit Random Forrest Classifier and optimize hyperparameters"""
from sklearn.ensemble import RandomForestClassifier
def rf_model_builder(model_params, model_dir):
  sklearn_model = RandomForestClassifier(**model_params)
  return dc.models.SklearnModel(sklearn_model, model_dir='models')

params_dict = {
def rf_model_builder(model_params, model_dir):
  sklearn_model = RandomForestClassifier(**model_params)
  return dc.models.SklearnModel(sklearn_model, model_dir='models')

params_dict = {
    "n_estimators": [10, 50, 100, 250, 500],
    "max_features": ["auto", "sqrt", "log2", None],
}
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
optimizer = dc.hyper.GridHyperparamOpt(rf_model_builder)
optimizer = dc.hyper.HyperparamOpt(rf_model_builder)
best_rf, best_rf_hyperparams, all_rf_results = optimizer.hyperparam_search(
    params_dict, train_dataset, valid_dataset, transformers,
    metric=metric)
best_rf.save()
pred = best_rf.predict(tmp_dataset)
pred.shape
pred_df = pd.DataFrame(pred)
pred_df.head
"""Fit and optimize Deep Neural Network"""
import numpy as np
params_dict = {"learning_rate": np.power(10., np.random.uniform(-5, -3, size=1))
, "weight_decay_penalty": np.power(10, np.random.uniform(-6, -4, size=1)),"nb_epoch": [40] }
n_features = train_dataset.get_data_shape()[0]
def model_builder(model_params, model_dir):
  model = dc.models.MultitaskClassifier(
    1, n_features, layer_sizes=[1000], dropouts=.25,
    batch_size=50, **model_params)
  return model

optimizer = dc.hyper.HyperparamOpt(model_builder)
best_dnn, best_dnn_hyperparams, all_dnn_results = optimizer.hyperparam_search(
    params_dict, train_dataset, valid_dataset, transformers,
    metric=metric)
best_rf.save()
from deepchem.utils.evaluate import Evaluator
rf_train_csv_out = "rf_train_classifier.csv"
rf_train_stats_out = "rf_train_stats_classifier.txt"
rf_train_evaluator = Evaluator(best_rf, train_dataset, transformers)
rf_train_score = rf_train_evaluator.compute_model_performance(
    [metric], rf_train_csv_out, rf_train_stats_out)
print("RF Train set AUC %f" % (rf_train_score["roc_auc_score"]))
rf_valid_csv_out = "rf_valid_classifier.csv"
rf_valid_stats_out = "rf_valid_stats_classifier.txt"
rf_valid_evaluator = Evaluator(best_rf, valid_dataset, transformers)
rf_valid_score = rf_valid_evaluator.compute_model_performance(
    [metric], rf_valid_csv_out, rf_valid_stats_out)
print("RF Valid set AUC %f" % (rf_valid_score["roc_auc_score"]))
 dnn_train_csv_out = "dnn_train_classifier.csv"
dnn_train_csv_out = "dnn_train_classifier.csv"
dnn_train_stats_out = "dnn_train_classifier_stats.txt"
dnn_train_evaluator = Evaluator(best_dnn, train_dataset, transformers)
dnn_train_score = dnn_train_evaluator.compute_model_performance(
    [metric], dnn_train_csv_out, dnn_train_stats_out)
print("DNN Train set AUC %f" % (dnn_train_score["roc_auc_score"]))
dnn_valid_csv_out = "dnn_valid_classifier.csv"
dnn_valid_stats_out = "dnn_valid_classifier_stats.txt"
dnn_valid_evaluator = Evaluator(best_dnn, valid_dataset, transformers)
dnn_valid_score = dnn_valid_evaluator.compute_model_performance(
    [metric], dnn_valid_csv_out, dnn_valid_stats_out)
print("DNN Valid set AUC %f" % (dnn_valid_score["roc_auc_score"]))
import readline
readline.write_history_file('my_py_script.txt')
readline.write_history_file('my_py_script.py')
