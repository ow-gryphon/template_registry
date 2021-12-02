import inspect
from functools import wraps

from labskit.utilities.errors import catch_errors
from labskit.utilities.log_configuration import configure_log
from labskit.settings import Settings
from labskit.storage.persist import SessionStore


def app(func):
    """
    Decorator function for configuring a new application. Automatically injects the right
    settings profile as selected from the commandline

    Usage::
       @app
       def my_analysis(settings, outputs):

          input_data = DataSource(settings).data
          transformed = input_data.pipe(filter_columns)
          outputs.save_pandas('clean_data', transformed)
          ...

    :param func: The main function to decorate
    :return: decorated function
    """
    settings = Settings()
    func_signature = inspect.signature(func)
    local_log = configure_log(settings).getChild('labskit')
    outputs = SessionStore(settings)

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        local_log.info("Executing {}".format(func.__name__))
        bound_args = func_signature.bind_partial(*args, **kwargs)

        binding = {
            'settings': settings,
            'outputs': outputs
        }

        final_args = inject_settings(func_signature, bound_args, binding)
        return func(*final_args.args, **final_args.kwargs)

    return catch_errors(wrapped_func)


def inject_settings(signature, bound_args, bindings):
    updated_kwargs = dict(bound_args.kwargs)
    for key, value in bindings.items():
        if (key in signature.parameters) and (key not in bound_args.arguments):
            updated_kwargs[key] = value

    return signature.bind(*bound_args.args, **updated_kwargs)
