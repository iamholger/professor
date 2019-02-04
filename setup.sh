#!/bin/bash

fpath=`readlink -f ${BASH_SOURCE[0]}`

profdir=`dirname ${fpath}`

PYV=`python -c "import sys;v=sys.version_info[0:2];print '%i.%i'%(v)"`

export LD_LIBRARY_PATH=${profdir}/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${profdir}/local/lib/python${PYV}/site-packages:$PYTHONPATH
export PYTHONPATH=${profdir}/local/lib64/python${PYV}/site-packages:$PYTHONPATH
export PATH=${profdir}/local/bin:$PATH
export PATH=${profdir}/local/contrib:$PATH
