#!/usr/bin/env python

import sys, os

sys.path.append(os.getcwd())

def main():
    assert sys.argv[1] in ('TPfracStep') #, 'NSfracStep', 'NSCoupled')
    solver = sys.argv.pop(1)
    if solver == 'TPfracStep':
        from twoasis import TPfracStep
    #elif solver == 'NSfracStep':
    #    from twoasis import NSfracStep
    #elif solver == 'NSCoupled':
    #    from twoasis import NSCoupled
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
