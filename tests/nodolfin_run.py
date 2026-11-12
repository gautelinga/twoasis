"""Run the twoasis command with legacy FEniCS made unimportable: an import of dolfin,
ufl_legacy, ffc, dijitso or FIAT raises, naming the line that tried. For the test that a
run on DOLFINx needs none of them (tests/test_backend_choice.py).

    python3 tests/nodolfin_run.py TPfracStep backend=dolfinx problem=... key=value ...
"""
import importlib.abc
import sys
import traceback

LEGACY = ("dolfin", "ufl_legacy", "ffc", "dijitso", "FIAT")


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in LEGACY:
            frames = [f for f in traceback.extract_stack()[:-1] if "importlib" not in f.filename]
            raise ImportError("legacy FEniCS imported on DOLFINx: {} from {}:{}".format(
                name, frames[-1].filename, frames[-1].lineno))
        return None


if __name__ == "__main__":
    sys.meta_path.insert(0, _Block())
    sys.argv = ["twoasis"] + sys.argv[1:]
    from twoasis.run_twoasis import main
    sys.exit(main())
