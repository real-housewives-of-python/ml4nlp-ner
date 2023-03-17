
This repository provides scripts and data files for the Machine Learning for NLP course.

### CODE ###


Here, you will find the following scripts which need to be executed in the following order:

- analyze_distribution.py 
- analyze_informative_features.py
- feature_extraction.py
- ner_system.py
- evaluation.py 
- feature_ablation.py
- crf.py
- crf_evaluation.py  



-------analyze_distribution.py ----------
Outputs as TSV file containing the distribution of the labels in the ConLL data
Instructions: run from commandline as analyze_distribution.py [path]
[path] = path to the file (train or development set) on your local machine

-------analyze_informative_features.py ----------
Outputs a TSV file with potentially informative features for data exploration
Instructions: run from commandline as python analyze_informative_features.py [path]
[path] = path to the file (ConLL train or development set) on your machine


----------feature_extraction.py -------------
Outputs a TSV file containing one-hot encoded features for each token extracted from a ConLL file
Instructions: run from commandline as python feature_extraction.py [path] 
[path] = path to the file (ConLL train, development or test set) on your machine


----------ner_system.py -------------
Performs NERC through Logistic Regression, SVM and Naive Bayes models; outputs ConLL file mapping predicted labels to gold labels 
Instructions: run from commandline as python ner_system.py [train path] [dev\test_path] [model] [outputfile] [embeddings] 
[train path] = path to the TSV training file generated from feature_extraction.py
[dev\test_path]  = path to the TSV dev or test file generated from feature_extraction.py
[model] = name of the model
[outputfile] = path and name of new ConLL file
[embeddings] True/False if you want to train the model on word embedding  


----------evaluation.py -------------
Provides confusion matrix, macro average and Precision, Recall and F1 score for each class without use of external modules.
Instructions: run from commandline as python evaluation.py [path] 
[path] = path to the ConLL file generated from ner_system.py


----------feature_ablation.py -------------
Performs a feature ablation study on Naive Bayes and Logistic Regression and combines word embeddings with one-hot features
Instructions: run from commandline as python feature_ablation.py  [train path] [dev path] [model] [embeddings] 
[train path] = path to the TSV training file generated from feature_extraction.py
[dev path]  = path to the TSV dev file generated from feature_extraction.py
[model] = name of the model
[embeddings] True/False if you want to combine the word embeddings with one-hot features


--------crf.py ----------
Performs NERC through Conditional Random Fields model; outputs TSV file mapping tokens to predicted labels
Instructions: run from commandline as python crf.py  [train path][dev/test path][outputfile]
[train path] = path to ConLL training set file
[dev/test path] = path to ConLL development or test set file
[outputfile] = path and name of new TSV file

----------crf_evaluation.py -------------
Provides confusion matrix, macro average and Precision, Recall and F1 score for each class without use of external modules.
Instructions: run from commandline as crf_evaluation.py [path] [dev/test path]
[path] = path to the TSV file with predicted labels generated from crf.py
[dev/test path] = path to ConLL development or test set file with the gold labels




### DATA ###

The data folder contains all outputs from the abovementioned scripts. 
