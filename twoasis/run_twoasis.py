#!/usr/bin/env python

import sys, os

sys.path.append(os.getcwd())

def main():
    """The twoasis command. Under MPI, an exception on one process ends the whole
    job (femtools.abort) instead of leaving the others waiting."""
    from femtools.abort import run
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
