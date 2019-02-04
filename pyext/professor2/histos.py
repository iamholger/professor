# -*- python -*-




class Histo(object):
    "A simple histogram -- just a Bin container with an optional path name"

    __slots__ = ["_bins", "path"]

    def __init__(self, bins=None, path=None):
        self._bins = bins if bins else []
        self.path = path

    @property
    def nbins(self):
        return len(self.bins)

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, bs):
        self._bins = bs
        self._bins.sort()
        for i, b in enumerate(self._bins):
            b.n = i

    def __len__(self):
        return self.nbins

    def __repr__(self):
        return "<%s with %d bins>" % (self.__class__.__name__, self.nbins)


class DataHisto(Histo):
    "Specialisation of Histo as a container of DataBins"

    def __init__(self, dbins=None, path=None):
        Histo.__init__(self, dbins, path)


    def toScatter2D(self, manpath=None):
        path = manpath if manpath is not None else self.path
        points = [(self.bins[i].xmid, self.bins[i].val,
                   [self.bins[i].xmid-self.bins[i].xmin, self.bins[i].xmax-self.bins[i].xmid],
                   self.bins[i].errs) for i in xrange(len(self.bins))]
        import yoda
        return yoda.Scatter2D(points, path, path)

    # def mkSmearedCopy(self):
        # Yorig    = [b.val        for b in self.bins]
        # from copy import deepcopy
        # B=deepcopy(self.bins)
        # Ysmeared = [b.smearBin() for b in B]
        # dY = [0.5*abs(o - s) for o, s in zip(Yorig, Ysmeared)]
        # import math
        # errorfactor=1.#math.sqrt(2.)
        # # Yerrnew = [b.err*errorfactor for b in B]
        # Yerrnew = [math.sqrt(b.err*b.err + dy*dy) for b, dy in zip(B, dY)]
        # # Yerrnew = [b.err + dy for b, dy in zip(B, dY)]
        # for num, b in enumerate(B):
            # b.val=Ysmeared[num]
            # b.errs=Yerrnew[num]
        # return Histo(B, self.path)

    def mkSmearedCopy(self, errorfactor=1):
        Yorig    = [b.val        for b in self.bins]
        NORM = sum(Yorig)
        from copy import deepcopy
        B=deepcopy(self.bins)
        Ysmeared = [b.smearBin() for b in B]
        NORMsmeared = sum(Ysmeared)
        # from IPython import embed
        # embed()
        # import sys
        # sys.exit(1)
        # import math
        Yerrnew = [b.err*errorfactor for b in B]
        # Yerrnew = [math.sqrt(b.err*b.err + dy*dy) for b, dy in zip(B, dY)]
        # Yerrnew = [b.err + dy for b, dy in zip(B, dY)]
        for num, b in enumerate(B):
            b.val=Ysmeared[num]*NORM/NORMsmeared
            # b.errs=Yerrnew[num]
        return Histo(B, self.path)

class IpolHisto(Histo):
    "Specialisation of Histo as a container of IpolBins"

    def __init__(self, ibins=None, path=None):
        Histo.__init__(self, ibins, path)

    def toDataHisto(self, *params):
        "Convert this IpolBin to a DataBin with values and errors computed at params"
        dbins = [ib.toDataBin(*params) for ib in self.bins]
        dhist = DataHisto(dbins, self.path)
        return dhist


class Bin(object):
    "A base class for binned data, handling the common x-edge stuff"

    __slots__ = ["xmin", "xmax", "n"]

    def __init__(self, xmin, xmax, n=None):
        self.xmin = xmin
        self.xmax = xmax
        self.n = n

    @property
    def xmid(self):
        return (self.xmin + self.xmax) / 2.0

    @property
    def xedges(self):
        return (self.xmin, self.xmax)

    def __cmp__(self, other):
        return cmp(self.xmin, other.xmin)

    def smear(self, SEED):
        pass


class DataBin(Bin):
    "A bin containing a data value and its error(s)"

    __slots__ = ["val", "_errs"]

    def __init__(self, xmin, xmax, val=None, errs=None):
        Bin.__init__(self, xmin, xmax)
        self.val = val
        self._errs = errs

    # TODO: return numerical 0 if _errs is None?
    @property
    def err(self):
        "Get a scalar error value, by averaging if necessary"
        if self._errs is None:
            return 0.0
        elif hasattr(self._errs, "__len__"):
            assert len(self._errs) == 2
            return sum(self._errs) / 2.0
        return self._errs
    @err.setter
    def err(self, e):
        "Set a scalar error value"
        assert not hasattr(self._errs, "__len__")
        self._errs = e

    @property
    def errs(self):
        "Get a pair of error values, by construction if necessary"
        if self._errs is None:
            return (0.0, 0.0)
        elif hasattr(self._errs, "__len__"):
            assert len(self._errs) == 2
            return self._errs
        return (self._errs, self._errs)
    @errs.setter
    def errs(self, e):
        "Set a pair of error values"
        if e is None:
            self._errs = None
        elif hasattr(e, "__len__"):
            assert len(e) == 2
            self._errs = e
        else:
            self._errs = [e,e]

    def smearBin(self):
        import random
        return random.gauss(self.val, self.err)

    def __repr__(self):
        return "<%s x=[%.3g..%.3g], y=%.3g, ey=[%.3g,%.3g]>" % \
            (self.__class__.__name__, self.xmin, self.xmax, self.val, self.errs[0], self.errs[1])


class IpolBin(Bin):
    """
    A bin containing a value interpolation and its error(s)

    TODO:
     * Provide ierr and ierrs getter/setter pairs cf. err/errs on DataBin? They can't be averaged (?), so not sure it makes sense...
     * Allow ipol'd error handling, with wrapped relative error parameterisation as an option?
    """

    __slots__ = ["ival", "ierrs", "__dict__"]

    def __init__(self, xmin, xmax, ival=None, ierrs=None, n=None):
        Bin.__init__(self, xmin, xmax, n)
        self.ival = ival
        self.ierrs = ierrs

    #def val(self, *params, vmin=None, vmax=None): #< needs Python3
    #    return self.ival.value(*params, vmin=vmin, vmax=vmax)
    def val(self, *params, **vminmax):
        "Get the interpolated value of this bin"
        vmin = vminmax.get("vmin", None)
        vmax = vminmax.get("vmax", None)
        return self.ival.value(*params, vmin=vmin, vmax=vmax)

    def der(self, *params, **vminmax):
        "Get the derivative according to the parametrisation"
        return self.ival.der(*params)

    def grad(self, *params, **vminmax):
        "Get the gradient according to the parametrisation"
        return self.ival.grad(*params)

    # TODO: need uniform access to two ipols regardless of what's stored (and maybe also a way to make a single avg Ipol...)

    #def err(self, *params, emin=0, emax=None): #< needs Python3
    def err(self, *params, **eminmax):
        "Get a single, averaged interpolated error for this bin"
        # emin = eminmax.get("emin", 0)
        # emax = eminmax.get("emax", None)
        # if self.ierrs is None:
        #     return 0.0
        # elif hasattr(self.ierrs, "__len__"):
        #     assert len(self.ierrs) == 2
        #     return (self.ierrs[0].value(*params, vmin=emin, vmax=emax) + self.ierrs[1].value(*params, vmin=emin, vmax=emax))/2.0
        # else:
        #     return self.ierrs.value(*params, vmin=emin, vmax=emax)
        es = self.errs(*params, **eminmax)
        return (es[0] + es[1])/2.0

    #def errs(self, params, emin=0, emax=None): #< needs Python3
    def errs(self, *params, **eminmax):
        "Get a pair of interpolated errors for this bin"
        emin = eminmax.get("emin", 0)
        emax = eminmax.get("emax", None)
        if self.ierrs is None:
            return (0.0, 0.0)
        elif hasattr(self.ierrs, "__len__"):
            assert len(self.ierrs) == 2
            return (self.ierrs[0].value(*params, vmin=emin, vmax=emax), self.ierrs[1].value(*params, vmin=emin, vmax=emax))
        else:
            e = self.ierrs.value(*params, vmin=emin, vmax=emax)
            return (e, e)

    @property
    def has_const_err():
        "Determine whether this bin's errors are fixed or variable -- the latter requires regularisation"
        if self.ierrs is None:
            return True
        if hasattr(self.ierrs, "__len__"):
            assert len(self.ierrs) == 2
            return (self.ierrs[0].order == 0 and self.ierrs[1].order == 0)
        else:
            return self.ierrs.order == 0

    #def toDataBin(self, *params, vmin=None, vmax=None, emin=0, emax=None): #< needs Python3
    def toDataBin(self, *params, **veminmax): #< needs Python3
        "Convert this IpolBin to a DataBin with values and errors computed at params, with optional range limits"
        vmin = veminmax.get("vmin", None)
        vmax = veminmax.get("vmax", None)
        emin = veminmax.get("vmin", 0)
        emax = veminmax.get("vmax", None)
        db = DataBin(self.xmin, self.xmax,
                     val=self.val(*params, vmin=vmin, vmax=vmax),
                     errs=self.errs(*params, vmin=emin, vmax=emax))
        return db

    def __repr__(self):
        s = "<%s x=[%.3g..%.3g]" % (self.__class__.__name__, self.xmin, self.xmax)
        try:
            s += ", %d params, ival order %d" % (self.ival.dim, self.ival.order)
        except:
            pass
        try:
            s += ", ierr order %d" % self.ierrs[0].order
        except:
            pass
        s += ">"
        return s



def mkEnvelope(histos):
    """ Take DataHistos and return coordinates for plotting """
    # Iterate over bins, get envelope representation
    # For each bin, return [xmin, xmax, mean(y), min(y), max(y)]
    E=[]
    for num, b in enumerate(histos[histos.keys()[0]].bins):
        t_b = [x.bins[num].val for x in histos.values()]
        E.append([b.xmin, b.xmax, sum(t_b)/len(t_b), min(t_b), max(t_b)])
    return E
