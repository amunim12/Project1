import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    if error_detail is not None:
        _,_,exc_tb=error_detail.exc_info()
        filename = exc_tb.tb_frame.f_code.co_filename
        error_message = "Error occured in python script name [{0}] at line number [{1}] error message [{2}]".format(
        filename, exc_tb.tb_lineno, str(error))
    else:
        error_message = str(error)
    return error_message

class customException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message