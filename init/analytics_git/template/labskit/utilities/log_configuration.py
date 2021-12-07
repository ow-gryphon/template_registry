import datetime
import logging

from labskit.settings import Settings


def configure_log(settings: Settings):
    """
    Configures the logger to have standard OW patterns.

    :param settings: Project settings
    :return: logging object for the source directory
    """
    log_file = settings.log_dir / "log_{key}.log".format(
        key=datetime.datetime.now().strftime("%Y%m%d-%H%M"))

    format_string = "[%(levelname)s] %(name)s: %(asctime)s - %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=format_string,
        filename=log_file
    )

    console_output = logging.StreamHandler()
    console_output.setLevel(logging.INFO)
    console_output.setFormatter(logging.Formatter(format_string))
    logging.root.addHandler(console_output)

    return logging.getLogger('src')
