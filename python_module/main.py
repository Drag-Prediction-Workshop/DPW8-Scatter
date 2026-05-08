"""USAGE:
Connect to an open instance of Tecplot or comment out tp.session.connect() to use batch.
Use command load_all() or load_single_datafile(path) to open data.

Use commands "" to verify data contains all aux values.

Analysis:
use the functions listed in __all__ to do any needed analysis. 
Aux attribute functions use tecplot constants.
Search https://tecplot.azureedge.net/products/pytecplot/docs/index.html if needed. 
see examples below.
Use reset_view() to return to single frame, initially loaded data.
Use universal_parser(..., verbose=True) to check ascii .dat veracity.

Example: Only show linemaps where SolverName == "CFD++ 18.1.0"
enable_linemaps_by_aux(
    aux_key="SolverName",
    aux_value="CFD++ 18.1.0",
    match_mode="equals",
    disable_other_linemaps=True)

Example: Set symbol shape to "gradient" for GridGroup == "dpw3"
set_linemap_attribute_by_aux(
    aux_key="GridGroup",
    aux_value="dpw3",
    attribute_path="symbols.shape",
    value="gradient",
    match_mode="equals")
"""

import tecplot as tp

from exporting import *
from load_data import *
from parsing import *
from plot_adjustments import *
from reformat_curves import *


__all__ = [
    "set_linemaps_axis_variable",
    "set_linemap_attribute_by_aux",
    "enable_linemaps_by_aux",
    "enable_linemaps_for_zone",
    "plot_linemap_diff",
    "plot_active_linemaps_diff",
    "reset_view",
    "universal_parser",
    "save_png",
]

tp.session.connect()
reset_view()
universal_parser()

enable_linemaps_for_zone(zones_containing_string="002.01")
plot_active_linemaps_diff()