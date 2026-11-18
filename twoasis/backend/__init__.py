"""The finite-element backend twoasis's driver and solvers run on.

The driver (TPfracStep.py), the solvers (solvers/TPfracStep) and the problems take their
finite-element names from here: femtools.backend's names (femtools.backend.NAMES), from
femtools.backend.dolfin or femtools.backend.dolfinx, with the dual mesh's measures and
Dirichlet conditions (twoasis/backend/_dual.py).

Which one is fixed when this module is first imported (`choose`): backend=dolfin,
backend=dolfinx or backend=auto on the command line, else the environment variable
TWOASIS_BACKEND, else auto. auto takes the backend a restart's checkpoint recorded
(params.dat), and dolfin otherwise.
"""
import os
import sys

BACKENDS = ("dolfin", "dolfinx")
DEFAULT = "dolfin"


def _argument(argv, key):
    """The value of key=value in argv (quotes stripped), or None."""
    value = None
    for arg in argv:
        if arg.startswith(key + "="):
            value = arg.split("=", 1)[1].strip().strip("'\"")
    return value


def checkpoint_backend(restart_folder):
    """The backend recorded in restart_folder/params.dat, or None (no file, or a
    checkpoint without the record, which is dolfin's)."""
    import pickle
    try:
        with open(os.path.join(restart_folder, "params.dat"), "rb") as f:
            return pickle.load(f).get("backend")
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError,
            IndexError, TypeError, ValueError):     # what unpickling a stray file raises
        return None


def choose(argv=None, environ=None):
    """The backend to run on, from the command line (argv), the environment and, for
    auto, the checkpoint of a restart."""
    argv = sys.argv[1:] if argv is None else argv
    environ = os.environ if environ is None else environ
    requested = _argument(argv, "backend") or environ.get("TWOASIS_BACKEND") or "auto"
    if requested == "auto":
        restart_folder = _argument(argv, "restart_folder")
        found = checkpoint_backend(restart_folder) if restart_folder else None
        requested = found if found in BACKENDS else DEFAULT
    if requested not in BACKENDS:
        raise ValueError("backend={!r}: one of {}, or auto".format(requested, ", ".join(BACKENDS)))
    return requested


def _names():
    import importlib
    from femtools.backend import NAMES
    from twoasis.backend._dual import dirichlet_class, measure_class
    base = importlib.import_module("femtools.backend." + choose())
    names = {n: getattr(base, n) for n in NAMES}
    names["Measure"] = measure_class(base.ufl)
    names["dx"] = names["Measure"]("dx")
    names["DirichletBC"] = dirichlet_class(base.DirichletBC, base.space_mesh)
    return names


globals().update(_names())
from femtools.backend import NAMES  # noqa: E402
__all__ = list(NAMES)
