import matplotlib as mpl
from cycler import cycler

# Defines the sequence of OW Colors
OW_COLOR_PALETTE = [
    '#348ABD',
    '#7A68A6',
    '#A60628',
    '#467821',
    '#CF4457',
    '#188487',
    '#E24A33'
]


def set_ow_colors():
    """
    Simple function to set the default color cycling for MatPlotlib to the 'ow standard'

    Usage::
       from labskit.utilities.ow_colors import set_ow_colors
       set_ow_colors()

       plt.plot([0, 1], [0, 1]) # This figure will have the line colors according
        to the OW Color palette

    """
    cyl = cycler('color', OW_COLOR_PALETTE)
    mpl.rcParams['axes.prop_cycle'] = cyl
    mpl.rcParams["patch.facecolor"] = OW_COLOR_PALETTE[0]
