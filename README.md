# CapstoneProject
Final project for Machine Learning Nanodegree programme.

In this project several classifiers for complex word identification for Swedish are trained and tested.

The dataset used in the training is CWI dataset for Swedish: it consists of more than 4,000 Swedish lemmas with annotated 39 linguistic features. All these features are already normalised and turned to their numerical representation.

Further reading about the dataset: https://pdfs.semanticscholar.org/8728/f63b7a08b1c9668bef101ba36a7950aa2432.pdf?_ga=2.200084270.1262024743.1596358064-547328618.1595177024

#Classification 

The repository contains two main Python modules: 

- train_random_forest.py: implements a random Forest classifier
- train_svm.py: implements a SVM classifier

In addition, the training set is added (train.svm)

Both modules contain two main menthods:
- a method for training a benchmark model using 3 features
- method for training a refined model using the full set of features

In addition to these methods, the modules do the necessary job for loading and preparing the dataset for the training as well as evaluating it and printing out the results.

#Running
The two modules can be run as Python scripts from the command line or using the main method in the modules.



