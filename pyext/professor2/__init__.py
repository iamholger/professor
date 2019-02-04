import sys
pyversion = sys.version_info
if sys.version_info[0] < 2 or (sys.version_info[0] == 2 and sys.version_info[1] < 7):
    raise Exception("Professor2 requires Python 2.7 or greater")

## Import Cython wrappings on the C++ core library
from professor2.core import *
__version__ = version()

from professor2.errors import *
from professor2.dataio import *
from professor2.histos import *
from professor2.ipolio import *
from professor2.ipol import *
from professor2.minimize import *
from professor2.paramsio import *
from professor2.params import *
from professor2.sampling import * #< the only module which requires NumPy
from professor2.weights import *
from professor2.chi2 import *
from professor2.eigentunes import *

def mk_timestamp():
    """
    Time stamp, taken from http://stackoverflow.com/questions/13890935/timestamp-python
    """
    import time
    ts = time.time()
    import datetime
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return st

# def mk_versionstring():
#     """
#     Get version from setup.py, courtesy of
#     http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
#     """
#     import pkg_resources  # part of setuptools
#     return pkg_resources.require("professor2")[0].version
# __version__ = mk_versionstring()


logo = \
"""
Visit us on http://professor.hepforge.org/
Please cite arXiv:0907.2973 [hep-ph]
                                          %s
 ______           __                            _____ _____
 | ___ \         / _|                          |_   _|_   _|
 | |_/ / __ ___ | |_ ___  ___ ___  ___  _ __     | |   | |
 |  __/ '__/ _ \|  _/ _ \/ __/ __|/ _ \| '__|    | |   | |
 | |  | | | (_) | ||  __/\__ \__ \ (_) | |      _| |_ _| |_
 |_|  |_|  \___/|_| \___||___/___/\___/|_|     |_____|_____|

 Andy Buckley, Holger Schulz                          v%s
 Copyright 2015-2017

""" % (mk_timestamp(), version())
