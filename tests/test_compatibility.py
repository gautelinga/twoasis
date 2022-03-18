import pytest
import subprocess
import re

########
# These tests only check if the untested code runs without breaking, and does not
# test any properties. These should be replaced with better tests.
########


@pytest.mark.skip(reason="Time")
@pytest.mark.parametrize("num_p", [1, 2])
@pytest.mark.parametrize("solver", ["IPCS", "Naive"])
@pytest.mark.parametrize("problem", ["Porous2D", "PlaneCouette"])
def test_demo_TPfracStep(num_p, solver, problem):
    cmd = "mpirun -np {} twoasis TPfracStep solver={} T=0.0001 dt=0.00005 problem={}"
    subprocess.check_output(cmd.format(num_p, solver, problem), shell=True)

if __name__ == "__main__":
    test_demo_TPfracStep()
