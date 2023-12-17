#!/usr/bin/env python

import sys, os
from twoasis.common.io import merge_visualization_files

def main():
    assert len(sys.argv) > 1
    folder = sys.argv[1]

    merge_visualization_files(folder)


if __name__ == "__main__":
    main()