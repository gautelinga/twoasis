#!/usr/bin/env python

from setuptools import setup

# Version number
major = 2018
minor = 1

setup(name = "twoasis",
      version = "%d.%d" % (major, minor),
      description = "Twoasis - Two-phase Navier-Stokes solvers in FEniCS",
      author = "Mikael Mortensen",
      author_email = "mikaem@math.uio.no",
      url = 'https://github.com/mikaem/Oasis.git',
      classifiers = [
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python ',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages = ["twoasis",
                  "twoasis.problems",
                  "twoasis.problems.TPfracStep",
                  #"twoasis.problems.NSfracStep",
                  #"twoasis.problems.NSCoupled",
                  "twoasis.solvers",
                  "twoasis.solvers.TPfracStep",
                  #"twoasis.solvers.NSfracStep",
                  #"twoasis.solvers.NSfracStep.LES",
                  #"twoasis.solvers.NSfracStep.NNModel",
                  #"twoasis.solvers.NSCoupled",
                  "twoasis.common",
                  ],
      package_dir = {"twoasis": "twoasis"},
      entry_points = {'console_scripts': ['twoasis=twoasis.run_twoasis:main']},
    )
