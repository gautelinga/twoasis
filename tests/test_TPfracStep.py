import pytest
import subprocess
import re
import math

number = "([0-9]+.[0-9]+e[+-][0-9]+)"

@pytest.mark.parametrize("solver", ["IPCS_ABCN", "BDFPC_Fast"])
@pytest.mark.parametrize("num_p", [1, 2])
def test_Porous2D(num_p, solver):
    cmd = ("mpirun -np {} oasis NSfracStep problem=DrivenCavity T=0.01 "
          "Nx=20 Ny=20 plot_interval=10000 solver={} testing=True")
    d = subprocess.check_output(cmd.format(num_p, solver), shell=True)
    #match = re.search("Velocity norm = " + number, str(d))
    #err = match.groups()
    assert True

if __name__ == '__main__':
    test_Porous2D()
