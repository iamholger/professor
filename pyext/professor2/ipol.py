# -*- python -*-

from __future__ import division
from professor2.core import *
from professor2.histos import *
from professor2.errors import *
from professor2.sampling import *


def mk_ipolinputs(params):
    """
    Make sorted run name and parameter name & value lists, suitable for passing to prof.Ipol

    params is a dict (actually, prefer OrderedDict) of run_names -> param_vals,
    as returned from read_rundata
    """
    runs = sorted(params.keys())
    if not runs:
        return runs, [], [[]]
    paramnames = params[runs[0]].keys()
    paramslist = [[params[run][pn] for pn in paramnames] for run in runs]
    return runs, paramnames, paramslist


def mk_ipolbin(rawP, rawV, rawE, xmin, xmax, CFG):
    # TODO finally learn how to use kwargs
    order    = CFG["ORDER"]
    errorder = CFG["ERR_ORDER"]
    errmode  = CFG["ERR_MODE"]
    medfilt  = CFG["MEDIAN_FILT"]

    if medfilt>0:
        from numpy import median
        # TODO figure out what to do with x=0
        relErrs = [rawE[num]/x if x!=0 else 1 for num, x in enumerate(rawV)]
        rem = medfilt*median(relErrs)
        P, V, E =[], [], []

        for num, x in enumerate(relErrs):
            if x < rem:
                P.append(rawP[num])
                V.append(rawV[num])
                E.append(rawE[num])
        if CFG["DEBUG"]:
            print "%i/%i survive median filter %f times %f"%(len(P), len(rawP), medfilt, median(relErrs))
    else:
        P=rawP
        V=rawV
        E=rawE
    import professor2 as prof
    pmin = prof.mk_minvals(P)
    pmax = prof.mk_maxvals(P)

    if order == "auto":
        valipol =mk_autoipol(P, V, CFG)
    else:
        valipol = Ipol(P, V, int(order))

    ## Check for NaN coeffs
    import math
    if any([math.isnan(x) for x in valipol.coeffs]):
        raise NanError("NaN coefficient encountered in value ipol")

    ## Build the error interpolation(s)
    if not errmode or errmode == "none":
        erripols = None
    ## Build the error interpolation(s)
    elif errmode == "mean":
        meanerr = sum(E) / float(len(E)) #histos[run].bins[binnr].err for run in runs) / float(len(runs))
        erripols = Ipol(P, [meanerr], 0) #< const 0th order interpolation
    elif errmode == "median":
        medianerr = E[len(E)//2]
        erripols = Ipol(P, [medianerr], 0) #< const 0th order interpolation
    elif errmode == "symm":
        if errorder == "auto":
            erripols = mk_autoipol(P, E, CFG)
        else:
            erripols = Ipol(P, E, int(errorder))
    elif errmode == "asymm":
        raise Exception("Error interpolation mode 'asymm' not yet supported")
    else:
        raise Exception("Unknown error interpolation mode '%s'" % errmode)

    if erripols is not None:
        if any([math.isnan(x) for x in erripols.coeffs]):
            raise NanError("NaN coefficient encountered in error ipol")

    return IpolBin(xmin, xmax, valipol, erripols), pmin, pmax

def mk_autoipol(P, V, CFG):
    omin  = CFG["AUTO_OMIN"]  if CFG.has_key("AUTO_OMIN")  else 0
    omax  = CFG["AUTO_OMAX"]  if CFG.has_key("AUTO_OMAX")  else 99
    split = CFG["AUTO_SPLIT"] if CFG.has_key("AUTO_SPLIT") else 0.1
    nit   = CFG["AUTO_NIT"]   if CFG.has_key("AUTO_NIT")   else 10
    debug = CFG["DEBUG"]      if CFG.has_key("DEBUG")      else False

    # Incides of all inputs --- needed for shuffeling
    ALLI = range(len(P))

    # Number of test points
    NTEST=int(split*len(ALLI))
    NTRAIN=len(ALLI)-NTEST

    # Prepare training samples
    trainings = [r for r in xrandomUniqueCombinations(ALLI, len(ALLI)-NTEST, nit)]
    # Prepare test samples
    tests = [[a for a in ALLI if not a in t] for t in trainings]

    # Dimension of parameter space
    DIM=len(P[0])

    # Get possible orders
    ORDERS=[]
    o_temp=omin
    while True:
        n_temp = numCoeffs(DIM, o_temp)
        if n_temp > NTRAIN or o_temp > omax:
            break
        ORDERS.append(o_temp)
        o_temp+=1

    residuals, meanresiduals, meanresidualstimesncoeff = {}, {}, {}
    for o in ORDERS:
        residuals[o] = []
        # Iterate through training "samples"
        for num, train in enumerate(trainings):
            # Calculate ipol for this run combination
            thisP = [P[x] for x in train]
            thisV = [V[x] for x in train]
            thisI = Ipol(thisP, thisV, o)
            # Get the residuals for all test points
            thisRes = [(thisI.val(P[x]) - V[x])**2 for x in xrange(len(tests[num]))]
            residuals[o].extend(thisRes)

    from numpy import mean
    for k, v in residuals.iteritems():
        meanresiduals[k] = mean(v)
        meanresidualstimesncoeff[k] = mean(v) * numCoeffs(DIM, k)
    #winner=min(meanresiduals, key=meanresiduals.get)
    winner=min(meanresidualstimesncoeff, key=meanresidualstimesncoeff.get)
    if debug:
        print "Residual summary:"
        print "Choose order %i"%winner
        for k, v in meanresiduals.iteritems():
            print "%i: %e times %i coeffs = %e"%(k, v, numCoeffs(DIM, k), meanresidualstimesncoeff[k])
    return Ipol(P, V, winner)


## Keep this for backward compatibility
def mk_ipolhisto(histos, runs, paramslist, order, errmode=None, errorder=None):
    """\
    Make a prof.IpolHisto from a dict of prof.DataHistos and the corresponding
    runs and params lists, at the given polynomial order.

    If errs is non-null, the data histo errors will also be interpolated.

    If errmode is None or 'none', uncertainties will not be parameterised and
    will return 0 if queried; 'mean' and 'median' will use fixed values derived
    from the anchor points; 'symm' will parameterise the average of the + and -
    errors of each bin at the polynomial order given by errorder. If errorder is
    None, the same order as for the value parameterisation will be used.

    Parameter range scaling will be applied, so a DoParamScaling=true flag will
    need to be written to the metadata when persisting the resulting IpolHisto.

    """
    if errmode is None:
        errmode = "none"
    if errorder is None:
        errorder = order
    #
    nbins = len(histos.itervalues().next().bins)
    ibins = []
    for n in xrange(nbins):
        ## Check that the bin edges are consistent and extract their values
        # TODO: move bin edge consistency checking into the Histo base class
        xmax = histos.values()[0].bins[n].xmax
        xmin = histos.values()[0].bins[n].xmin
        vals = [histos[run].bins[n].val for run in runs]
        errs = [histos[run].bins[n].err for run in runs]
        try:
            ibins.append(mk_ipolbin(paramslist, vals, errs, xmin, xmax, order, errmode, errorder))
        except NanError, ne:
            print ne, "in bin %i of %s" % (n, histos.values()[0].path)
    return Histo(ibins, histos.values()[0].path)


# https://stackoverflow.com/questions/2130016/splitting-a-list-of-into-n-parts-of-approximately-equal-length
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
      out.append(seq[int(last):int(last + avg)])
      last += avg
    return out


def mkStandardIpols(HISTOS, HNAMES, RUNS, PARAMSLIST, CFG, nchunks=10, referenceIpolSet=None, quiet=False):

    """
    the referenceIpolSet allows to make structurally identical ipols
    """
    BNAMES = []
    for hn in HNAMES:
        histos = HISTOS[hn]
        nbins = histos.values()[0].nbins
        for n in xrange(nbins):
            BNAMES.append([hn, n])
            if referenceIpolSet is not None:
                b_order_val = referenceIpolSet[hn].bins[n].ival.order
                b_order_err = referenceIpolSet[hn].bins[n].ierrs.order
                BNAMES[-1].append(b_order_val)
                BNAMES[-1].append(b_order_err)

    NBINS = len(BNAMES)

    MSGEVERY = int(NBINS/100.) if NBINS > 100 else 1;

    import sys, zlib
    import professor2 as prof
    def worker(q, rdict, counter):
        "Function to make bin ipols and store ipol persistency strings for each histo"
        import sys
        while True:
            if q.empty():
                break
            try:
                temp = q.get(False)
            except:
                break

            hn = temp[0]
            histos = HISTOS[hn]
            n = temp[1]
            xmax = histos.values()[0].bins[n].xmax
            xmin = histos.values()[0].bins[n].xmin
            from math import log
            if CFG["LOGY"]:
                vals = [log(histos[run].bins[n].val) if histos[run].bins[n].val>0 else  0  for run in RUNS]
                errs = [log(histos[run].bins[n].err) if histos[run].bins[n].err>0 else -1  for run in RUNS]

            else:
                vals = [histos[run].bins[n].val for run in RUNS]
                errs = [histos[run].bins[n].err for run in RUNS]
            try:
                if referenceIpolSet is not None:
                    CFG["ORDER"] = temp[2]
                    CFG["ERR_ORDER"]=temp[3]
                    ib, pmin, pmax = prof.mk_ipolbin(PARAMSLIST, vals, errs, xmin, xmax, CFG)
                else:
                    ib, pmin, pmax = prof.mk_ipolbin(PARAMSLIST, vals, errs, xmin, xmax, CFG)
                s = ""
                s += "%s#%d %.5e %.5e\n" % (hn, n, ib.xmin, ib.xmax)
                s += "  " + ib.ival.toString("val")
                for v in pmin:
                    s+= " %.5e"%v
                for v in pmax:
                    s+= " %.5e"%v
                s+= "\n"
                if ib.ierrs:
                    s += "  " + ib.ierrs.toString("err")
                    for v in pmin:
                        s+= " %.5e"%v
                    for v in pmax:
                        s+= " %.5e"%v
                    s+= "\n"
                rdict.put( [hn,n, zlib.compress(s, 9)])
                del s
                del ib #< pro-actively clear up memory
            except NanError, ne:
                print ne, "in bin %i of %s" % (n, histos.values()[0].path)
            del histos
            counter.value += 1
            if counter.value == MSGEVERY and not quiet:
                counter.value = 0
                sys.stderr.write('\rProgress: {current}/{total}\r'.format(current=rdict.qsize(), total=NBINS))
            q.task_done()
        return

    # TODO: Printing and multiprocessing should happen under script control
    if not CFG["QUIET"]:
        print "\nParametrising %i objects...\n" % len(BNAMES)
    import time, multiprocessing
    rDict = {}
    time1 = time.time()

    from multiprocessing import Manager, Process
    manager = Manager()

    # This for the status --- modulus is too expensive
    ndone = manager.Value('i', 0)
    ## A shared memory object is required for coefficient retrieval
    r = manager.Queue()

    # # For testing with IPython embed --- leave the following 4 lines in please
    # q = manager.Queue()
    # for chunk in chunkIt(BNAMES, nchunks): # The chunking is necessary as the memory blows up otherwise
        # map(lambda x:q.put(x), chunk)
        # worker(q, r , ndone)

    for chunk in chunkIt(BNAMES, nchunks): # The chunking is necessary as the memory blows up otherwise

        ## The job queue
        q = manager.Queue()

        ## Fire away
        workers = [Process(target=worker, args=(q, r, ndone)) for i in xrange(CFG["MULTI"])]
        map(lambda x:q.put(x), chunk)
        map(lambda x:x.start(), workers)

        map(lambda x:x.join(),  workers)
        map(lambda x:x.terminate(),  workers)

    ## Timing
    while not r.empty():
        a, b, c = r.get()
        rDict[(a,b)] =c

    time2 = time.time()
    if not CFG["QUIET"]:
        print('\n\nParametrisation took %0.2fs' % ((time2-time1)))

    return rDict


def writeIpol(fname, ipolDict, params, runs=[], summary="", runsdir=""):
    PARAMNAMES = params[0]
    PARAMSLIST = params[1]

    import os, tempfile, zlib
    if fname == "temp":
        f = tempfile.NamedTemporaryFile(delete=False)
    else:
        f = open(fname, "w")

    import professor2 as prof
    f.write("Summary: %s\n" % summary)
    f.write("DataDir: %s\n" % os.path.abspath(runsdir))
    f.write("ProfVersion: %s\n" % prof.version())
    f.write("Date: %s\n" % prof.mk_timestamp())
    f.write("DataFormat: binned 3\n") # This tells the reader how to treat the coefficients that follow
    # Format and write out parameter names
    pstring = "ParamNames:"
    for p in PARAMNAMES:
        pstring += " %s" % p
    f.write(pstring + "\n")
    # Dimension (consistency check)
    f.write("Dimension: %i\n" % len(PARAMNAMES))
    # Interpolation validity (hypercube edges)
    # TODO a lot of this is obsolete in the format 3
    minstring = "MinParamVals:"
    for v in prof.mk_minvals(PARAMSLIST):
        minstring += " %f" % v
    f.write(minstring + "\n")
    maxstring = "MaxParamVals:"
    for v in prof.mk_maxvals(PARAMSLIST):
        maxstring += " %f" % v
    f.write(maxstring + "\n")
    f.write("DoParamScaling: 1\n")
    # Number of inputs per bin
    f.write("NumInputs: %i\n" % len(PARAMSLIST))
    s_runs = "Runs:"
    for r in runs:
        s_runs +=" %s"%r
    f.write("%s\n"%s_runs)
    f.write("---\n")

    ## Write out numerical data for all interpolations
    s = ""
    HNAMES=sorted(list(set([x[0] for x in ipolDict.keys()])))
    for hn in sorted(HNAMES):
        thisbins=sorted(filter(lambda x: x[0]==hn, ipolDict.keys()))
        for ipolstring in [ipolDict[x] for x in thisbins]:
            s+=zlib.decompress(ipolstring)

    f.write(s)
    f.close()
    if not fname == "temp":
        print "\nOutput written to %s" % fname
    else:
        return f




def getBox(P, masterbox, mastercenter, debug=False):
    """
    Helper function that finds the most appropriate box for a parameter point P
    """
    import professor2 as prof
    boxdict=None
    for box, bdict in masterbox.iteritems():
        if prof.pInBOX(P, box, debug):
            boxdict = bdict
            break
    if boxdict is None:
        distances={}
        for c in mastercenter.keys():
            distances[prof.pBoxDistance(P, c)] = c
        winner = min(distances.keys())
        boxdict = mastercenter[distances[winner]]
    return boxdict
