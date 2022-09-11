import os
import sys

class AdultCensusException(Exception):
    
    def __init__(self, error_message:Exception,error_detail:sys):
        super().__init__(error_message)
        self.error_message=AdultCensusException.get_deatiled_error_message(error_message =error_message,
                                                                       error_detail=error_detail)


    @staticmethod
    def get_deatiled_error_message(error_message:Exception,error_detail:sys)->str:
        _,_ ,exec_tb = error_detail.exc_info()
        """
        error_message: Exception object
        error_detail: object of sys module
        """

        line_number = exec_tb.tb_frame.f_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename
        error_message = f"Error occured in scrip: [{file_name}] at line number: [{line_number}] error message: [{error_message}]"
        return error_message


    def __str__(self):
        return self.error_message
        

    def __repr__(self) -> str:
       return AdultCensusException.__name__.str()