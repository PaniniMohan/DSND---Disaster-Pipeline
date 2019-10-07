# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python ./data/process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python ./models/train_classifier.py ./DisasterResponse.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. How to see the APP -
	- If you are working in Udacity workspace : Kindly follow this link : https://knowledge.udacity.com/questions/20899
	- If you are running in the local machine : Go to http://0.0.0.0:3001/


#### Packages needed to work -

- import sys

#### Libraries related to Load_Data function
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
import pickle
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import fbeta_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

#### Other dependencies

Python 3.5+ (I used Python 3.7)
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
Natural Language Process Libraries: NLTK
SQLlite Database Libraqries: SQLalchemy
Web App and Data Visualization: Flask, Plotly

#### Acknowledgements

Udacity Course on machinelearning pipeline
Figure Eight for providing messages dataset to train my model

