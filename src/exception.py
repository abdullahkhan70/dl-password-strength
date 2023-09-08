import sys
from dataclasses import dataclass


def error_message_detail(message, error_detail: sys):
    _, _, exe_tb = error_detail.exc_info()
    file_name = exe_tb.tb_frame.f_code.co_filename
    line_no = exe_tb.tb_lineno
    error_message = f"Error occured i Script Name: {file_name} \
        \n Line Number: {line_no}\n Error Message: {str(message)}"

    return error_message


@dataclass
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys) -> None:
        super().__init__()
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
