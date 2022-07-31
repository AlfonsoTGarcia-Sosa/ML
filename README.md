# ML

Code and jupyter notebooks to run analysis and classification as in paper García-Sosa A. T.* "Androgen Receptor Binding Category Prediction with Deep Neural Networks and Structure-, Ligand-, and Statistically-Based Features" Molecules 2021, 26:1285 (https://www.mdpi.com/1420-3049/26/5/1285/htm)

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

roc_hum_dud.png, roc_chimp_dud.png, roc_rat_dud.png are ROC-AUC curves for Human, Chimp, and Rat AR docking with DUD-E created decoys. Respectively, AUC values are 0.80, 0.83, and 0.74. Enrichment factor EF(1%) for Chimp AR = 68.92


Also consult publications:
Peña-Guerrero J., Nguewa P.,* García-Sosa A. T.* "Machine Learning, Artificial Intelligence, and Data Science Breaking into Drug Design and Neglected Diseases" WIREs Computational Molecular Science 2021, 11(5):e1513 (https://doi.org/10.1002/wcms.1513) 

Yosipof A., Guedes R. C., García-Sosa A. T.* "Data Mining and Machine Learning Models for Predicting Drug Likeness and their Disease or Organ Category" Frontiers in Chemistry 2018, 6:162 (https://doi.org/10.3389/fchem.2018.00162) 
Highlighted in the Specialty News Technology Networks "Data Visualization in Biopharma: Leveraging AI, VR, and MR to Support Drug Discovery" https://www.technologynetworks.com/drug-discovery/articles/data-visualization-in-biopharma-leveraging-ai-vr-and-mr-to-support-drug-discovery-320108?fbclid=IwAR2mdhtKtxrObLLG3GqsLouHN3dwwGK_tvHcukvqakm5FNJYZBeWaO3q1Y8

García-Sosa A. T.* "Benford's Law in Medicinal Chemistry: Implications for Drug Design" Future Medicinal Chemistry 2019, 11(17):2247-2253 (https://doi.org/10.4155/fmc-2019-0006) 

