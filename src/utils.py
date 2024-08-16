import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import customException

def save_obj(file_path, obj):
    '''
    This function is responsible for saving the object in the given file path.
    '''
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise customException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        model_report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)


            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            model_report[list(models.keys())[i]] = test_model_score

        return model_report

    except Exception as e:
        raise customException(e, sys)