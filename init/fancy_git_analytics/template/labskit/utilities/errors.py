import logging
from functools import wraps


def catch_errors(func):
    """
    Decorator function for logging errors that occur during execution.
    That is, if we are executing a function and it errors out, we catch the exception here,
    log it to the associated log-file, and re-raise the exception so as to ensure that we
    properly interrupt execution.


    :param func: function to wrap
    :return: wrapped function

    Usage:

        @log_errors
        def an_error_prone_function(args):
            raise Exception("this will cause problems")

        When `an_error_prone_function` is called, it will raise an error that will subsequently
        be written to the log before continuing.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        log = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception as excepted:
            # catch any generic exception
            log.error("Error in executing {fnc}".format(fnc=func.__name__))
            log.exception(excepted)
            # raise excepted

    return wrapped
