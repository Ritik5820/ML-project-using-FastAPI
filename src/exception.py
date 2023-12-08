import sys


def error_message_detail(error, error_detail: sys):
    """Format error details into a readable message.

    Args:
        error (Exception): The exception object representing the error.
        error_detail (sys): The `sys` module information containing details about the error.

    Returns:
        str: A formatted error message including script name, line number, and error details.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """Initialize the CustomException with an error message and details.

        Args:
            error_message (str): The main error message.
            error_detail (sys): The `sys` module information containing details about the error.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        """Return a string representation of the exception.

        Returns:
            str: A formatted error message including script name, line number, and error details.
        """
        return self.error_message
