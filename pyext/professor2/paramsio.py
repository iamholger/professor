# -*- python -*-

from professor2.utils import mkdict as _mkdict

def read_paramsfile(path):
    """
    Read a file with parameters
    """
    rtn = _mkdict()
    with open(path, "r") as f:
        L = [l.strip() for l in f if not l.startswith("#")]
        for num, line in enumerate(L):
            parts = line.split()
            if len(parts) == 2:
                rtn[parts[0]] = float(parts[1])
            elif len(parts) == 1:
                rtn["PARAM%i" % num] = float(parts[0])
            else:
                print "Error in parameter input format"
                import sys
                sys.exit(1)
    return rtn

def read_limitsandfixed(fname):
    """
    Read a text file e.g.
    PARAM1  0         1   # interpreted as fixed param
    PARAM2  0.54444       # interpreted as limits
    """
    limits, fixed = {}, {}
    if fname is not None:
        with open(fname) as f:
            for l in f:
                if not l.startswith("#"):
                    temp = l.split()
                    if len(temp) == 2:
                        fixed[temp[0]] = float(temp[1])
                    elif len(temp) == 3:
                        limits[temp[0]] = (float(temp[1]), float(temp[2]))
    return limits, fixed
