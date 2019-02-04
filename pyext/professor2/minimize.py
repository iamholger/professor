# -*- python -*-

def mk_fitfunc(fname, pnames, globname, extraargs=[]):
    """
    Dynamically make a fit function for the given param names, to be passed to Minuit.

    Return a string definition of the function, to be exec'd, and the list of
    generated internal arg names corresponding to pnames.
    """
    fargs = ["A%03i" % i for i in xrange(len(pnames))]
    funcdef = "def {gname}({fargs}):\n    x= lambda {fargs}: {fname}([{fargs}], {extrargs})".format(gname=globname, fargs=", ".join(fargs), fname=fname, extrargs=", ".join(extraargs))
    funcdef+="\n    return x({fargs})".format(fargs=", ".join(fargs))
    return funcdef

def mk_classfitfunc(fname, pnames, globname, extraargs=[]):
    """
    Dynamically make a fit function for the given param names, to be passed to Minuit.

    Return a string definition of the function, to be exec'd, and the list of
    generated internal arg names corresponding to pnames.
    """
    fargs = ["A%03i" % i for i in xrange(len(pnames))]
    funcdef = "def {gname}(self, {fargs}):\n    x= lambda {fargs}: {fname}([{fargs}], {extrargs})".format(gname=globname, fargs=", ".join(fargs), fname=fname, extrargs=", ".join(extraargs))
    funcdef+="\n    return x({fargs})".format(fargs=", ".join(fargs))
    return funcdef

def prepareBox(ipolFile, dataHistos, matchers=None, maxErrDict=None, doFilter=False, debug=False):#, normWeights=False):
    import professor2 as prof
    ## Read interpolated and reference histos, and run data
    IHISTOS, METADATA = prof.read_ipoldata(ipolFile)
    PNAMES = METADATA["ParamNames"].split()
    if not PNAMES:
        PNAMES = ["A%03i" % i for i in xrange(int(METADATA["Dimension"]))]
    PMIN   = [float(x) for x in METADATA["MinParamVals"].split()]
    PMAX   = [float(x) for x in METADATA["MaxParamVals"].split()]

    # Prepare tuples to be used as keys in ipol box dicts
    center = tuple([PMIN[i] + 0.5*(PMAX[i]-PMIN[i]) for i in xrange(len(PMIN))])
    box    = tuple([(PMIN[i], PMAX[i])              for i in xrange(len(PMIN))])

    ## Find objects available in both ipol and ref data (and in matchers if not None)
    available = []
    for ihn in sorted(IHISTOS.keys()):
        ## Set default bin weights
        for ib in IHISTOS[ihn].bins:
            ib.w = 1.0
        ## Find user-specified bin weights if there was a weight file
        if matchers is not None:
            ## Find matches
            pathmatch_matchers = [(m,wstr) for m,wstr in matchers.iteritems() if m.match_path(ihn)]
            ## Ditch histos not listed in the weight file
            if not pathmatch_matchers:
                del IHISTOS[ihn]
                continue
            ## Attach fit weights to the ibins, setting to zero if there's no position match
            for ib in IHISTOS[ihn].bins:
                posmatch_matchers = [(m,wstr) for (m,wstr) in pathmatch_matchers if m.match_pos(ib)]
                ib.w = float(posmatch_matchers[-1][1]) if posmatch_matchers else 0 #< NB. using last match
        for rhn in dataHistos.keys():
            if ihn==rhn or rhn=="/REF/"+ihn: #< TODO: short for rhn = "/REF/"+ihn ?
                # TODO: we should eliminate this potential mismatch of ref and MC hnames
                available.append([ihn,rhn])
                break #< TODO: ok? NOT SURE

    ## Prepare lists of ibins and dbins
    IBINS, DBINS, MAXERRS, FILTERED = [], [], [], []
    BINDICES={} # Allows for more helpful error messages in case of prof.StatError
    for a in available:
        if len(IHISTOS[a[0]].bins) != len(dataHistos[a[1]].bins):
            print "Inconsistency discovered between data bins and parametrised bins:"
            print "Removing histogram", a[0]
            del IHISTOS[a[0]]
            del dataHistos[a[1]]
        else:
            BINDICES[a[0]] = []#range(len(IBINS),  len(IBINS) +     len(IHISTOS[a[0]])) # This is for debugging
            for nb in xrange(len(IHISTOS[a[0]].bins)):
                if doFilter and dataHistos[a[1]].bins[nb].err == 0:
                    FILTERED.append(1)
                    continue
                if IHISTOS[a[0]].bins[nb].w >0:
                    IBINS.append(IHISTOS[a[0]].bins[nb])
                    DBINS.append(dataHistos[a[1]].bins[nb])
                    BINDICES[a[0]].append(len(IBINS)-1)
            if maxErrDict:
                MAXERRS.extend(MAXERRS[a[0]])
    if debug:
        print "Using the following objects:"
        for a in available:
            print "IpolObject", a[0], "DataObject", a[1]
    if not MAXERRS:
        MAXERRS = None

    if debug:
        print "DEBUG: filtered %i bins due to zero data error" % len(FILTERED)

    ## Sanity checks
    assert len(IBINS) == len(DBINS)
    if not IBINS:
        raise prof.NoBinsError("No bins left ... exiting")

    assert MAXERRS is None or len(IBINS) == len(MAXERRS)

    PMIN = map(float, METADATA["MinParamVals"].split())
    PMAX = map(float, METADATA["MaxParamVals"].split())
    center = tuple([PMIN[i] + 0.5*(PMAX[i]-PMIN[i]) for i in xrange(len(PMIN))])
    box    = tuple([(PMIN[i], PMAX[i])              for i in xrange(len(PMIN))])
    returntuple = (
            ("IBINS",    IBINS),
            ("DBINS",    DBINS),
            ("BINDICES", BINDICES),
            ("MAXERRS",  MAXERRS),
            ("IHISTOS",  IHISTOS),
            ("PNAMES",   PNAMES),
            ("BOX",      box),
            ("CENTER",   center)
            )

    return box, center, returntuple

def boxFilt(boxdict, obsnames):
    """
    Make a sub set of a box for a list of observables
    """
    # TODO also use MAXERRS ???
    IBINS, DBINS = [], []
    MAXERRS=None
    BINDICES={}
    IHISTOS={}
    PNAMES = boxdict["PNAMES"]
    box    = boxdict["BOX"]
    center = boxdict["CENTER"]
    for k, v in boxdict["BINDICES"].items():
        if len(v) >0:
            if k in obsnames:
                BINDICES[k] = []
                for nb in v:
                    IBINS.append(boxdict["IBINS"][nb])
                    DBINS.append(boxdict["DBINS"][nb])
                    BINDICES[k].append(len(IBINS)-1)
                IHISTOS[k] = boxdict["IHISTOS"][k]

    return dict(
            (
            ("IBINS",    IBINS),
            ("DBINS",    DBINS),
            ("BINDICES", BINDICES),
            ("MAXERRS",  MAXERRS),
            ("IHISTOS",  IHISTOS),
            ("PNAMES",   PNAMES),
            ("BOX",      box),
            ("CENTER",   center)
            )
            )


def pInBOX(P, box, debug=False):
    for num, p in enumerate(P):
        if debug:
            print min(box[num]), "<", p, "<", max(box[num]), "?"
        if p<min(box[num]) or p>max(box[num]):
            return False
    return True

def pBoxDistance(A, B):
    import math
    return math.sqrt(sum([ (A[i]-B[i])*(A[i]-B[i]) for i in xrange(len(A))]))

def simpleGoF(params, masterbox, mastercenter, debug, unitweights=False):
    """
    Very straightforward goodness-of-fit measure
    """
    import professor2 as prof
    chi2 = 0.0
    # Get the right box first
    boxdict=None
    if len(masterbox.keys())==1:
        boxdict=masterbox.values()[0]
    else:
        for box, bdict in masterbox.iteritems():
            if pInBOX(params, box, debug):
                boxdict = bdict
                break
        if boxdict is None:
            distances={}
            for c in mastercenter.keys():
                distances[pBoxDistance(params, c)] = c
            winner = min(distances.keys())
            boxdict = mastercenter[distances[winner]]

    ibins=boxdict["IBINS"]
    dbins=boxdict["DBINS"]
    maxerrs=boxdict["MAXERRS"]
    bindices=boxdict["BINDICES"]
    for num, ibin in enumerate(ibins):
        ## Weight is attached to the ipol bin (default set to 1.0 above)
        w = ibin.w
        if w == 0:
            continue
        ## Get ipol & ref bin values and compute their difference
        ival = ibin.val(params)
        dval = dbins[num].val
        diff = dval - ival
        ## Data error
        err2 = dbins[num].err**2
        ## Plus interpolation error added in quadrature
        maxierr = maxerrs[ibin] if maxerrs else None
        err2 += ibin.err(params, emax=maxierr)**2
        # TODO: compute asymm error for appropriate deviation direction cf. sum([e**2 for e in ibin.ierrs])
        if not err2:
            culprit=""
            i_culprit=-1
            for k, v in bindices.iteritems():
                if num in v:
                    culprit=k
                    i_culprit = v.index(num)
            raise prof.StatError("Zero uncertainty on a bin being used in the fit -- cannot compute a reasonable GoF!\n\tObservable: %s\n\t%s %f+=%f\n\t%s \nSee weight-syntax in documentation or user --filter CL arg to remove bins with zero data error automatically"%(culprit, ibin, ival, ibin.err(params, emax=maxierr),  dbins[num]))
        # TODO: should we square w too, so it penalised deviations _linearly_?
        # NOTE yes!
        if unitweights:
            chi2 +=  diff**2 / err2
        else:
            chi2 += w * w * diff**2 / err2
    return chi2



def setupMinuitFitarg(pnames, pstart, pmins, pmaxs, limits, fixed, allowExtrapolation=False, verbose=True):
    ## Dictionary fitarg for iminuit
    farg=dict()

    ## Initial conditions --- use pos = center of hypercube, and step = range/10
    assert len(pmins) == len(pmaxs)

    pranges = [(pmaxs[i] - pmins[i]) for i in xrange(len(pmins))]

    # This sets the start point
    for i, aname in enumerate(pnames):
        farg[aname] = pstart[i]
        farg['error_%s'%aname] = pranges[i] / 10.


    for i, pname in enumerate(pnames):
        if not allowExtrapolation:
            if verbose:
                print "Setting minimiser limit to interpolation limit for '%s', you can use -x to allow extrapolation"%pname
            farg['limit_%s'%pname] = (pmins[i],pmaxs[i])
        if pname in limits.keys():
            farg['limit_%s'%pname] = limits[pname]
        if pname in fixed.keys():
            farg[pname] = fixed[pnames[i]]
            farg['fix_%s'%pname] = True
    return farg


def readResult(fname):
    """
       Open results file, extract and return minimum point as OrderedDict
       and return raw list of all other lines for further processing.
    """
    RES=[]
    OTH=[]
    with open(fname) as f:
        for line in f:
            l=line.strip()
            if l.startswith("#"):
                OTH.append(l)
            else:
                temp=l.split()
                RES.append([temp[0], float(temp[1])])

    from collections import OrderedDict
    return OrderedDict(RES), OTH

def getParamCov(TXT):
    """
       Read the covariance matrix from the lines, return as numpy array
    """
    START = TXT.index("# Covariance matrix:") + 2
    dim = len(TXT[START].strip().split()) - 2
    END = START+dim
    COV_raw = TXT[START:END]
    COV_txt = [COV_raw[d].split()[2:2+dim] for d in xrange(dim)]

    # Go through line by line and find fixed params
    fixed=[]
    for num, c in enumerate(COV_txt):
        if all([x=='---' for x in c]):
            fixed.append(num)

    # This is the reduced cov matrix
    COV_l = []
    for num, c in enumerate(COV_txt):
        if not num in fixed:
            cline = [float(x) for num2, x in enumerate(c) if not num2 in fixed]
            COV_l.append(cline)

    # Cast it into a numpy array
    from numpy import zeros
    COV_p = zeros((dim-len(fixed), dim-len(fixed)))
    for i in xrange(dim-len(fixed)):
        for j in xrange(dim-len(fixed)):
            COV_p[i][j] = COV_l[i][j]


    return COV_p

def getFixedParams(TXT):
    """
       Read fixed parameters
    """
    from collections import OrderedDict
    fixed=OrderedDict()
    START = TXT.index("# Fixed:") + 1
    for line in TXT[START:]:
        l=line[1:].strip().split()
        if len(l)==0:
            return fixed
        else:
            fixed[l[0]]=float(l[1])

def getParamLimits(TXT):
    """
       Read parameters limits
    """
    from collections import OrderedDict
    fixed=OrderedDict()
    START = TXT.index("# Limits:") + 1
    for line in TXT[START:]:
        l=line[1:].strip().split()
        if len(l)==0:
            return fixed
        else:
            fixed[l[0]]=(float(l[1]), float(l[2]))

def getParamCorr(TXT):
    """
       Read the corrlation matrix from the lines, return as numpy array
    """
    START = TXT.index("# Correlation matrix:") + 2
    dim = len(TXT[START].strip().split()) - 2
    END = START+dim
    COV_raw = TXT[START:END]
    from numpy import zeros
    COV_p = zeros((dim, dim))
    for i in xrange(dim):
        temp = map(float, COV_raw[i].split()[2:2+dim])
        for j in xrange(dim):
            COV_p[i][j] = temp[j]


    return COV_p

def eigenDecomposition(mat):
    """
    Given a symmetric, real NxN matrix, M, an eigen decomposition is always
    possible, such that M can be written as M = T_transp * S * T (orthogonal
    transformation) with T_transp * T = 1_N and S being a diagonal matrix with
    the eigenvalues of M on the diagonal.

    Returns
    -------
    T_trans : numpy.matrix
    S : numpy.ndarray
        The real eigen values.
    T : numpy.matrix
    """
    from scipy import linalg
    from numpy import matrix
    import numpy

    A = matrix(mat)
    S, T_trans = linalg.eig(A)
    if numpy.iscomplex(S).any():
        raise ValueError("Given matrix `mat` has complex eigenvalues!")

    return matrix(T_trans), S.real, matrix(T_trans).transpose()

def mkEigentunes(COV, point, plus=True):
    T_trans, S, T = eigenDecomposition(COV)
    from numpy import sqrt, zeros, matrix
    rv = matrix(point.values())
    rv_trans = (T_trans * rv.transpose()).transpose()

    ret = []

    for num, c in enumerate(S):
        ev = zeros(len(S))
        sigma=sqrt(c)
        if plus:
            ev[num] =sigma
        else:
            ev[num] = -1* sigma
        ev_trans = rv_trans + ev
        etune_params_t = T * ev_trans.transpose()
        etune_params = etune_params_t.transpose().tolist()[0]
        ret.append([sigma, etune_params])

    return ret
