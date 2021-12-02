from labskit.sources.local_csv import LocalCsv


class LocalMMC(LocalCsv):
    """
    Reads the sample csv data embedded in the the base Labskit Project
    """
    def __init__(self, settings):
        super().__init__(settings, "sample_data.csv")
