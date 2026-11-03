# Based on oasis/common/__init__.py from Oasis, Copyright (C) Mikael Mortensen
# <mikaem@math.uio.no>, GNU Lesser GPL version 3 or any later version.
# Modified for two-phase flow by Gaute Linga, 2022-2026.
from .io import (
    check_if_kill, create_initial_folders, init_from_restart,
    merge_visualization_files, merge_xml_files, save_checkpoint_solution_h5, save_solution,
    save_tstep_solution_h5)
from .utilities import (
    A_cache, assemble_matrix, AssignedVectorFunction, compute_ind, DivRHS,
    GradFunction, homogenize, Mat_cache_dict, OasisFunction,
    PhiGradRHS, TPsource)
import sys
import ast
import json

__all__ = [
    "parse_command_line",
    # io
    "check_if_kill", "create_initial_folders", "init_from_restart",
    "merge_visualization_files", "merge_xml_files", "save_checkpoint_solution_h5", "save_solution",
    "save_tstep_solution_h5",
    # utilities
    "A_cache", "assemble_matrix", "AssignedVectorFunction", "compute_ind",
    "DivRHS", "GradFunction", "homogenize", "Mat_cache_dict",
    "OasisFunction", "PhiGradRHS", "TPsource"]

# Parse command-line keyword arguments
def parse_command_line():
    commandline_kwargs = {}
    for s in sys.argv[1:]:
        if '=' not in s:
            raise TypeError("{}: arguments are key=value; the parameters are the defaults in "
                            "twoasis/problems/TPfracStep/__init__.py and those of the "
                            "problem file".format(s))
        key, value = s.split('=', 1)
        # JSON first (numbers, lists, dicts, true/false), then Python literals
        # (True/False, tuples); anything else stays a string.
        try:
            value = json.loads(value)
        except ValueError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass

        commandline_kwargs[key] = value
    return commandline_kwargs

# A dict on the command line updates a default dict, e.g.
#   twoasis TPfracStep velocity_update_solver='{"method": "gradient_matrix"}'
