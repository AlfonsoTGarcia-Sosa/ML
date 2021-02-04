# ML

my_py_scriptMoreMetrics.py contains Random Forest and Deep Neural Network classifiers and regressors on the CoMPARA set of androgen receptor (AR) toxicity compounds using my own features. It runs using environment file deepch.yml. First create a conda environment with:

conda env create -f deepch.yml

Then

conda activate deepch

Then

python

and run my_py_scriptMoreMetrics.py
such as:
python my_py_scriptMoreMetrics.py > out.txt



Old:<br>
balanced_myfeats*.ipynb contain Random Forest and Deep Neural Network classifiers and regressors on the CoMPARA set of androgen receptor toxicity compounds using my own features

Old:<br>
balanced_GraphConv*.ipynb contain Graph Convolutional Network classifiers and regressors on the CoMPARA set of androgen receptor toxicity compounds

Old:<br>

balanced_GraphConv*scores.ipynb contain Graph Convolutional Network classifiers and regressors on the CoMPARA set of androgen receptor toxicity compounds, as well as scoring extra set of molecules

The latest version of the notebooks have also text comments

roc_hum_dud.png, roc_chimp_dud.png, roc_rat_dud.png are ROC-AUC curves for Human, Chimp, and Rat AR docking with DUD-E created decoys. Respectively, AUC values are 0.80, 0.83, and 0.74. Enrichement factor EF(1%) for Chimp AR = 68.92
