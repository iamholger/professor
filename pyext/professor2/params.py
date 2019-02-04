# -*- python -*-


def scangrid(*args, **kwargs):
    """
    Generator function which returns lists of parameter (name, value)
    2-tuples via yields (i.e. in a for-expression).

    The arguments are 4-tuples of (name, numpts, low, high) for each
    parameter; the generator iterations will then visit each of the
    lattice points in a grid with num points between low..high in each
    param.
    """
    vec = kwargs.get("vec", [])
    if args:
        # print "@", len(args) #, args
        var, npts, low, high = args[0]
        for val in [low + i*(high-low)/(npts-1.) for i in xrange(npts)]:
            newargs = args[1:]
            newvec = vec + [(var, val)]
            newkwargs = kwargs
            newkwargs["vec"] = newvec
            for i in scangrid(*newargs, **newkwargs):
                yield i
            # TODO: in Python3 this can replace the extra for-loop above
            # yield from scan(*newargs, **newkwargs):
    else:
        yield vec


def mk_minvals(anchors):
    dim=len(anchors[0])
    mins=[]
    for i in xrange(dim):
        mins.append(min([a[i] for a in anchors]))
    return mins

def mk_maxvals(anchors):
    dim=len(anchors[0])
    maxs=[]
    for i in xrange(dim):
        maxs.append(max([a[i] for a in anchors]))
    return maxs

def mk_center(anchors):
    """
    Calculate center from all anchor points --- needed?
    Requires knowledge of all anchor points, hmm
    """
    mins = mk_minvals(anchors)
    maxs = mk_maxvals(anchors)
    center = []
    for num, m in enumerate(mins):
        center.append(m+0.5*(maxs[num]-m))
    return center

def is_inrange(ppoint, minv, maxv):
    dec = True
    notinrange = []
    for i in xrange(len(ppoint)):
        if ppoint[i] < minv[i] or ppoint[i] > maxv[i]:
            dec=False
            notinrange.append(i)
    return dec, notinrange
