import logging
import time
from functools import wraps


def time_function(func):
    """
    Decorator function that logs execution time for the wrapped function


    :param func: Function to time
    :return: Wrapped function

    Usage:
        @time_function
        def my_long_running_function(some_arg):
            do_something
            return something
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        log = logging.getLogger(func.__module__)
        start_time = time.time()
        func_name = func.__name__
        log.info("Beginning execution of {funcname}".format(funcname=func_name))
        results = func(*args, **kwargs)
        stop_time = time.time()
        log.info("Execution time of {func_name}: "
                 "{exectime:0.02f}".format(func_name=func_name,
                                           exectime=stop_time - start_time))
        return results

    return wrapped
