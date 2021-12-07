import contextlib
import datetime
import enum
import logging
import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tinydb import TinyDB, Query

from labskit.utilities.paths import get_path


class Content(enum.Enum):
    """
    Concise labeling for storage content types
    """
    DATAFRAME = 0
    MODEL = 1
    OUTPUTFILE = 2


class SessionStore:
    """
    Storage a session or batch of analysis. Establishes a unique folder where a set of files,
    dataframes, and analysis outputs can be saved and uniquely identified later. Use this to save
    and retrieve data results throughout an analysis script.

    Example::
       # build a new store and save a data frame
       output = SessionStore(settings)
       output.store_pandas('clean_data', cleaned_dataframe)

       ...
       # retrieve the output later
       clean_data = output.get_pandas('clean_data')
       clean_data.shape

    :param settings: A Project Settings class to manage general configuration
       and where to find many important folders
    :param run_id: an identifier for a specific through-run of the analysis
    :param scopes: A list of sub-scoping strings to provide hierarchical structure in the
       outputs
    """
    def __init__(self, settings, run_id=None, scopes=None):
        self.run_id = run_id or datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.settings = settings
        self.scopes = scopes or []

        base_path = settings.output / self.run_id
        self.storage_root = get_path(Path.joinpath(base_path, *self.scopes))
        self.db = TinyDB(self.storage_root / "metadata.json")

    def scope(self, subscope):
        """
        Builds a new SessionStore with a designated scope. Adding a new scope subname will put the
        outputs in a hierarchical folder structure. For example, with the scope
        "ridge_regression", all of the outputs will be stored in the folder
        "outputs/<runid>/ridge_regression".

        Scopes can be added sequentially, that is `outputs.scope('regressions').scope('ridge')`
        will save output in `outputs/<runid>/regressions/ridge`.

        :param subscope: name of additional scope
        :return: A new RunStore with additional scope hierarchy
        """
        return self.__class__(
            settings=self.settings,
            run_id=self.run_id,
            scopes=self.scopes + [subscope]
        )

    @property
    def _local_log(self):
        """
        Access to the logging instance for internal use in the session store methods
        :return: configured logger
        """
        local_log = logging.getLogger(__name__).getChild(self.run_id)
        return local_log

    def store_pandas(self, name, frame):
        """
        Stores a named dataframe in the output folder

        :param name: string name of a dataframe to store
        :param frame: the Pandas frame to store
        :return:
        """
        self._local_log.info("storing {} as parquet frame".format(name))
        cache_file = (self.storage_root / name).with_suffix(".parquet")

        with self._write_once(name):
            metadata = {
                "name": name,
                "filepath": str(cache_file),
                "content": Content.DATAFRAME.name
            }
            self.db.insert(metadata)

            pa_table = pa.Table.from_pandas(df=frame)
            pq.write_table(pa_table, cache_file)

    @contextlib.contextmanager
    def _write_once(self, name):
        """
        Safely provides access to a given file object for writing
        :param name: Name of object to retrieve
        :return:
        """
        records = self.db.search(Query().name == name)
        if any(records):
            raise KeyError("There is already an object at {}".format(name))
        yield

    def item_info(self, name):
        """
        Gets information about an item that may be stored
        :param name: name of the item to query
        :return: Dictionary of information about the stored item
        """
        return self.query_one(Query().name == name)

    def query_one(self, query):
        """
        Looks for a specific entry in the local database
        :param query: TinyDb Query to run
        :return: A single result from the database
        """
        results = self.db.search(query)
        if len(results) == 1:
            return results[0]
        elif len(results) == 0:
            return None
        else:
            raise KeyError("multiple keys for query {}".format(query))

    def get_frame(self, name):
        """
        Gets a dataframe from disk according to the name for this run-id

        :param name: Name of the frame to retrieve
        :return: Pandas dataframe read from disk
        """
        metadata = self.item_info(name)
        self._local_log.info("Reading {} from cache".format(metadata['filepath']))
        parquet = pq.read_table(metadata['filepath'])
        return parquet.to_pandas()

    def store_model(self, name, model):
        """
        Stores a model in the project storage using pickle to write the contents to a file.

        :param name: name of the model to store
        :param model: the model to store
        :return:
        """
        with self._write_once(name):
            filepath = (self.storage_root / name).with_suffix(".pkl")
            metadata = {
                "name": name,
                "filepath": str(filepath),
                "content": Content.MODEL.name
            }
            self.db.insert(metadata)
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)

    def get_model(self, name):
        """
        Retrieves a model with a given name. Assumes that the content stored was a pickle.

        :param name: Name of the model
        :return: Retrieved model
        """
        info = self.item_info(name)
        if info:
            filepath = info['filepath']
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            self._local_log.warning("File not found {}".format(name))

    def project_file(self, name):
        """
        File name for saving in this project. For example, to save a PNG output file or to
        specifically write a CSV output.

        :param name: Name of the file to link
        :return: The full filepath with output-specific folder structure
        """
        full_file = self.storage_root / name
        if not full_file.exists():
            return full_file

        else:
            self._local_log.warning("file already exists")
            return None
