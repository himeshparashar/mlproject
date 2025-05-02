import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(obj, file_path):
    """
    Save the object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:  
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate the given models and return a report of their performance.
    """
    try:
        report = {}

        for model_name, model in models.items():

            gs = GridSearchCV(
                estimator=model,
                param_grid=params[model_name],
                cv=3
            )
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            print(f"Best parameters for {model_name}: {gs.best_params_}")
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            report[model_name] = np.round(r2_score(y_test, y_test_pred), 2)
        return report

    except Exception as e:
        raise CustomException(e, sys) from e