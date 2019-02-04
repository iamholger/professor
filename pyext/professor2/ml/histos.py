# -*- python -*-

from professor2.histos import *

class MLHisto(Histo):
    "Specialisation of Histo as a container of MLBins"

    def __init__(self, nnbins=None, path=None):
        Histo.__init__(self, nnbins, path)

    def toDataHisto(self, *params):
        "Convert this IpolBin to a DataBin with values and errors computed at params"
        dbins = [ib.toDataBin(*params) for ib in self.bins]
        dhist = DataHisto(dbins, self.path)
        return dhist


class MLBin(Bin):
    """
    A bin containing a value Machine Learning
    """

    __slots__ = ["mlval", "__dict__"]

    def __init__(self, xmin, xmax, X, Y, pnames=None, pC=1.0, pEps=0.0):
        Bin.__init__(self, xmin, xmax)
        from numpy import array
        X=array(X)
        Y=array([[y,0] for y in Y]) # Manky hack to get rid of the deprecation warning

        # Data scaling --- Standard scaler works much better than MinMaxScaler
        from sklearn import preprocessing
        self._xscaler = preprocessing.StandardScaler()
        xscaled = self._xscaler.fit_transform(X)
        self._yscaler = preprocessing.StandardScaler()
        yscaled = self._yscaler.fit_transform(Y) # This produces the noisy deprecation warning

        # Machine Learning magic
        from sklearn import svm
        ml = svm.SVR(kernel='rbf', C=pC, epsilon=pEps) # TODO --- explore parameters of SVR
        ml.fit(xscaled, yscaled[:,0]) # PArt of the hack
        self.mlval = ml

    def val(self, *params):
        "Get the ML prediction  of this bin"
        from numpy import array
        p_raw =array(params[0][0]).reshape(1,-1)  # The raw, unscaled param point
        p_scaled = self._xscaler.transform(p_raw) # The scaled param point
        ret_raw = self.mlval.predict(p_scaled)    # The prediction in the scaled value world
        ret = self._yscaler.inverse_transform([ret_raw,0]) # The prediction in the unscaled value world
        return float(ret[0]) # part of the hack

    def toDataBin(self, *params): #< needs Python3
        "Convert this NNBin to a DataBin with values at params"
        db = DataBin(self.xmin, self.xmax,
                     val=self.val(params),
                     )
        return db

def mk_MLHisto(histos, runs, paramslist, paramnames=None, pC=1.0, pEps=0.0):
    from numpy import array

    nbins = len(histos.itervalues().next().bins)
    mbins = []

    for n in xrange(nbins):
        xmins = set([histos[run].bins[n].xmin for run in runs])
        xmaxs = set([histos[run].bins[n].xmax for run in runs])
        xmin, xmax = xmins.pop(), xmaxs.pop()
        vals = [histos[run].bins[n].val for run in runs]

        mbins.append(MLBin(xmin, xmax, array(paramslist), array(vals), paramnames, pC, pEps))

    return MLHisto(mbins, histos.values()[0].path)
