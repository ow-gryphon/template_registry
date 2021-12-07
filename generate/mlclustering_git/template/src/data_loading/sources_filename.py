from labskit.sources.web_cache import WebCsv


class HmdaNdSmall(WebCsv):
    """
    A sample set of HMDA data from the Home Mortgage Disclosure Act website
    https://www.consumerfinance.gov/data-research/hmda/explore
    """

    @property
    def simple_name(self):
        return "hmda_nd_small"

    @property
    def url(self):
        return "https://raw.githubusercontent.com/labskit/"\
            "analytics-sample-data/master/mlclustering/hmda_ND_small.csv"
