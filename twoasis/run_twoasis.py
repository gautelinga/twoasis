#!/usr/bin/env python

import sys, os

# One BLAS/OpenMP thread a process unless the environment says otherwise: MPI ranks
# already fill the cores, and threaded BLAS in every rank oversubscribes them. Set
# before numpy and PETSc load their BLAS.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.append(os.getcwd())

def main():
    """The twoasis command. Under MPI, an exception on one process ends the whole
    job (feluga.abort) instead of leaving the others waiting."""
    from feluga.abort import run
    return run(lambda argv: _main())

def _main():
    assert len(sys.argv) > 1
    assert sys.argv[1] in ('TPfracStep',)
    solver = sys.argv.pop(1)
    if solver == 'TPfracStep':
        from twoasis import TPfracStep
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
