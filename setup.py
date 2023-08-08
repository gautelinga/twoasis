#!/usr/bin/env python

from setuptools import setup

# Version number
major = 2023
minor = 1

setup(name = "twoasis",
      version = "%d.%d" % (major, minor),
      description = "Twoasis - Two-phase flow solvers in FEniCS",
      author = "Gaute Linga",
      author_email = "gaute.linga@mn.uio.no",
      url = 'https://github.com/gautelinga/twoasis.git',
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
      entry_points = {'console_scripts': ['twoasis=twoasis.run_twoasis:main', 'quickstats=twoasis.quickstats:main']},
    )
