# Based on oasis/common/__init__.py from Oasis, Copyright (C) Mikael Mortensen
# <mikaem@math.uio.no>, GNU Lesser GPL version 3 or any later version.
# Modified for two-phase flow by Gaute Linga, 2022-2026.
from .io import (
    check_if_kill, create_initial_folders, init_from_restart,
    merge_visualization_files, merge_xml_files, save_checkpoint_solution_h5, save_solution,
    save_tstep_solution_h5)
from .utilities import (
    A_cache, assemble_matrix, AssignedVectorFunction, compute_ind, DivFunction,
    GradFunction, homogenize, Mat_cache_dict, OasisFunction,
    PhiGradFunction, Solver_cache, Solver_cache_dict, TPsource)
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
    "DivFunction", "GradFunction", "homogenize", "Mat_cache_dict",
    "OasisFunction", "PhiGradFunction", "Solver_cache", "Solver_cache_dict", "TPsource"]

# Parse command-line keyword arguments
def parse_command_line():
    commandline_kwargs = {}
    for s in sys.argv[1:]:
        if s.count('=') == 1:
            key, value = s.split('=', 1)
        else:
            raise TypeError((s + " Only kwargs separated with '=' sign " +
                            "allowed. See NSdefault_hooks for a range of " +
                            "parameters. Your problem file should contain " +
                            "problem specific parameters."))
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

# Note to self. To change a dictionary variable through commandline do, e.g.,
# run NSfracStep velocity_update_solver='{"method":"gradient_matrix"}'
