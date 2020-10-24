# Wind Power Forecasting - Application tool

This repository contains the source code of my Final Master's degree project in [Decision Systems Engineering](https://www.urjc.es/estudios/master/915-ingenieria-de-sistemas-de-decision), titled *Wind Power Forecasting using Machine Learning techniques*, coursed in Rey Juan Carlos University. It is based on the [Data Science challenge](https://challengedata.ens.fr/participants/challenges/34/) posed by the *Compagnie nationale du Rhône* ([CNR](https://www.cnr.tm.fr/). For further information you can read the [dissertation](dissertation.pdf) (spanish).



## Introduction
This application is intented to be a flexible and configurable tool in order to easily build and analyze models for this forecasting problem. It is based on [Kedro](https://kedro.readthedocs.io/en/stable/index.html) API for the sake of applying software engineering best practices to data and machine-learning pipelines. [MLflow tracking](https://mlflow.org/) is used to record and query experiments (code, data, config, and results).

## Instalation

## Implemented pipelines and CLI commands

The main pipelines implemented are:
1. Prepare data for EDA (`eda`). Transforms raw data into a proper format for Exploratory Data Analisys.
2. Data engineering (`de`). Gets the data ready to be consumed by Machine Learning algorithms.
3. Feature engineering (`fe`). Allows to explore and add new features to the data sets.
4. Modeling (`mdl`). Trains the selected algorithm, optmizes hyperparameters of the model and make predictions on the test set.

There are other two additional pipelines:
1. CNR pipeline. It contains several subpipelines to get predictions and submission file for the CNR Data Science Challege.
2. Neural Networks. In progress ...



