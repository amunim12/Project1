import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle

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