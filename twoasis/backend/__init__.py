"""The finite-element backend twoasis's driver and solvers run on.

The driver (TPfracStep.py), the solvers (solvers/TPfracStep) and the projections
(common/utilities.py) take their finite-element names from here instead of from dolfin, so
that another backend can provide them: legacy dolfin (twoasis/backend/dolfin.py, whose
docstring lists the names: the contract) or DOLFINx (twoasis/backend/dolfinx.py).

Which one is fixed when this module is first imported (`choose`): backend=dolfin,
backend=dolfinx or backend=auto on the command line, else the environment variable
TWOASIS_BACKEND, else auto. auto takes the backend a restart's checkpoint recorded
(params.dat), and dolfin otherwise: dolfin stays the default for now.
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
    checkpoint from before backends were recorded, which is dolfin's)."""
    import pickle
    try:
        with open(os.path.join(restart_folder, "params.dat"), "rb") as f:
            return pickle.load(f).get("backend")
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError):
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


if choose() == "dolfin":
    from twoasis.backend.dolfin import *  # noqa: F401,F403
    from twoasis.backend.dolfin import __all__  # noqa: F401
else:
    from twoasis.backend.dolfinx import *  # noqa: F401,F403
    from twoasis.backend.dolfinx import __all__  # noqa: F401
