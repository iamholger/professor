# -*- python -*-

from professor2.errors import *
from professor2.ipol import *
from professor2.histos import *


class IpolMeta(dict):

    def __init__(self, ifile=None):
        if ifile:
            self.update(self.read_ipolmeta(ifile))

    @property
    def dim(self):
        return int(self.get("Dimension", -1))

    @property
    def pnames(self):
        return self.get("ParamNames", "").split()

    @property
    def pvalsmin(self):
        return [float(x) for x in self.get("MinParamVals", "").split()]

    @property
    def pvalsmax(self):
        return [float(x) for x in self.get("MaxParamVals", "").split()]

    @property
    def numinputs(self):
        return int(self.get("NumInputs", -1))

    def read_ipolmeta(self, ifile):
        """
        Read in meta data from prof-ipol output 'ifile'
        """
        meta = {}

        if type(ifile) == str:
            ifile=open(ifile)
        ifile.seek(0)
        for l in ifile.readlines():
            ## Strip out comments
            if "#" in l:
                l = l[:l.find("#")]
            ## Ignore blank / pure whitespace lines
            l = l.strip()
            if not l:
                continue
            ## Exit if we see the end-of-header indicator
            if l == "---":
                break
            ## Extract the key-value pair from the line
            try:
                key, value = [str.strip(s) for s in l.split(":", 1)]
                meta[key] = value
            except:
                print "Couldn't extract key-value pair from '%s'" % l
        return meta


def read_ipolmeta(ifile):
    return IpolMeta(ifile)


def read_simpleipols(ifile, paramlimits=None):
    """
    Read ipol data back in from ifile.

    If the paramlimits argument is non-null, it will be used internally by
    the Ipol objects to stabilise the SVD calculation. For this to make sense,
    the persisted ipols must have been created with the same scaling factors.
    paramlimits should be a 2-tuple of lists for min and max param values
    respectively.
    """
    IOBJECTS = {}
    if type(ifile) == str:
        ifile=open(ifile)
    ifile.seek(0)
    name = ""
    for line in ifile.readlines():
        sline = line.strip()
        if sline.startswith("/"):
            name = sline.split()[0]
        elif sline.startswith("val"):
            IOBJECTS[name] = Ipol(sline)
            if paramlimits:
                IOBJECTS[name].setParamLimits(*paramlimits)
    return IOBJECTS


def read_binnedipols(ifile, paramlimits=None):
    """
    Read binned ipol data back in from ifile.

    If the paramlimits argument is non-null, it will be used internally by
    the Ipol objects to stabilise the SVD calculation. For this to make sense,
    the persisted ipols must have been created with the same scaling factors.
    paramlimits should be a 2-tuple of lists for min and max param values
    respectively.
    """
    IHISTOS = {}
    if type(ifile) == str:
        ifile=open(ifile)
    ifile.seek(0)
    name = ""
    for line in ifile.readlines():
        sline = line.strip()
        if sline.startswith("/"):
            fullpath, sxmin, sxmax = sline.split()
            hpath, nbin = fullpath.split("#")
            currentib = IpolBin(float(sxmin), float(sxmax), n=int(nbin))
            IHISTOS.setdefault(hpath, IpolHisto(path=hpath)).bins.append(currentib)
        elif sline.startswith("val"):
            currentib.ival = Ipol(sline)
            if paramlimits:
                currentib.ival.setParamLimits(*paramlimits)
            #print currentib.ival.coeffs()
        elif sline.startswith("err"):
            currentib.ierrs = Ipol(sline)
            if paramlimits:
                currentib.ierrs.setParamLimits(*paramlimits)
            #print currentib.ierrs.coeffs()
        # TODO: read back asymm errs as two ipols
    return IHISTOS


def read_ipoldata(ifile):
    "Return both the metadata object and collection of IpolHistos from a binned ipol file"
    imeta = read_ipolmeta(ifile)
    if not imeta["DataFormat"].startswith('binned'):
        raise IpolIOError("Error, DataFormat of ipol file %s is not binned" % ifile)
    df = int(imeta["DataFormat"].split()[-1])
    if df<2:
        raise IpolIOError("Error, DataFormat '%s' of ipol file %s is not supported by this version of Professor, please recalculate parametrisations." %(imeta["DataFormat"] ,ifile))
    paramlimits = None
    if bool(int(imeta.get("DoParamScaling", 0))):
        if imeta["DataFormat"].endswith('2'):
            assert imeta.has_key("MinParamVals") and imeta.has_key("MaxParamVals")
            minparamvals = [float(s) for s in imeta["MinParamVals"].split()]
            maxparamvals = [float(s) for s in imeta["MaxParamVals"].split()]
            paramlimits = (minparamvals, maxparamvals)
        # Note, in format 3, the min max values are stored at the end of the coefficient vector
        # the Ipol.setParamLimits function is protected agains overwriting those limits
        # when reading in format 3
    return read_binnedipols(ifile, paramlimits), imeta
