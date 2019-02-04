# -*- python -*-

from professor2.histos import *
from professor2.paramsio import *
import os.path, glob


def read_histos_root(path):
    "Load histograms from a ROOT file, into a dict of path -> yoda.Histo[DataBin]"
    histos = {}

    # TODO: Could just use YODA for everything, including ROOT reading?
    try:
        import ROOT
        ROOT.gROOT.SetBatch(True)
    except ImportError:
        print "PyROOT not available... skipping"
        return histos

    # TODO: use yoda.root.getall
    def _getallrootobjs(d, basepath="/"):
        "Recurse through a ROOT file/dir and generate (path, obj) pairs"
        for key in d.GetListOfKeys():
            kname = key.GetName()
            if key.IsFolder():
                # TODO: -> "yield from" in Py3
                for i in _getallrootobjs(d.Get(kname), basepath+kname+"/"):
                    yield i
            else:
                yield basepath+kname, d.Get(kname)
    try:
        f = ROOT.TFile(path)
        for rname, robj in _getallrootobjs(f):
            bins = []
            if robj.InheritsFrom("TH1"):
                # TODO: allow 2D histos
                if robj.InheritsFrom("TH2"):
                    continue
                for ib in xrange(robj.GetNbinsX()):
                    xmin = robj.GetXaxis().GetBinLowEdge(ib+1)
                    xmax = robj.GetXaxis().GetBinUpEdge(ib+1)
                    y = robj.GetBinContent(ib+1)
                    ey = robj.GetBinError(ib+1)
                    bins.append(DataBin(xmin, xmax, y, ey))
                histos[rname] = Histo(bins, rname)
            elif robj.InheritsFrom("TGraph"):
                for ip in xrange(robj.GetN()):
                    x, y = ROOT.Double(), ROOT.Double()
                    robj.GetPoint(ip, x, y)
                    xmin = x - robj.GetErrorXlow(ip)
                    xmax = x + robj.GetErrorXhigh(ip)
                    ey = (robj.GetErrorXlow(ip) + robj.GetErrorXhigh(ip)) / 2.0
                    bins.append(DataBin(xmin, xmax, y, ey))
            histos[rname] = Histo(bins, rname)
        f.Close()
    except Exception, e:
        print "Can't load histos from ROOT file '%s': %s" % (path, e)

    return histos


def read_histos_yoda(path):
    "Load histograms from a YODA-supported file type, into a dict of path -> yoda.Histo[DataBin]"
    histos = {}
    try:
        import yoda
        s2s = []
        aos = yoda.read(path, asdict=False)
        for ao in aos:
            import os
            ## Skip the Rivet cross-section and event counter objects
            # TODO: Avoid Rivet-specific behaviour by try block handling & scatter.dim requirements
            if os.path.basename(ao.path).startswith("_"):
                continue
            ##
            s2s.append(ao.mkScatter())
        del aos #< pro-active YODA memory clean-up
        #
        for s2 in filter(lambda x:x.dim==2, s2s): # Filter for Scatter1D
            bins = [DataBin(p.xMin, p.xMax, p.y, p.yErrAvg) for p in s2.points]
            histos[s2.path] = Histo(bins, s2.path)
        del s2s #< pro-active YODA memory clean-up
    except Exception, e:
        print "Can't load histos from file '%s': %s" % (path, e)
    return histos


def read_histos(filepath, stripref=True):
    """
    Load histograms from file, into a dict of path -> yoda.Histo[DataBin]

    If stripref = True, remove any leading /REF prefix from the histo path
    before putting it in the dictionary.
    """
    histos = {}
    if filepath.endswith(".root"):
        histos.update(read_histos_root(filepath))
    elif any(filepath.endswith(ext) for ext in [".yoda", ".aida", ".flat"]):
        histos.update(read_histos_yoda(filepath))
    if stripref:
        newhistos = {}
        for p, h in histos.iteritems():
            if p.startswith("/REF"):
                p = p.replace("/REF", "", 1)
                h.path = p
            newhistos[p] = h
        histos = newhistos
    return histos


def read_all_histos(dirpath, stripref=True):
    """
    Load histograms from all files in the given dir, into a dict of path -> yoda.Histo[DataBin]

    If stripref = True, remove any leading /REF prefix from the histo path
    before putting it in the dictionary.
    """

    histos = {}
    filepaths = glob.glob(os.path.join(dirpath, "*"))
    for fp in filepaths:
        histos.update(read_histos(fp, stripref))
    return histos


def read_rundata(dirs, pfname="params.dat", verbosity=1): #, formats="yoda,root,aida,flat"):
    """
    Read interpolation anchor point data from a provided set of run directory paths.

    Returns a pair of dicts, the first mapping run names (i.e. rundir basenames) to
    the parameter value list for each run, and the second mapping observable names
    (i.e. histogram paths) to a run -> histo dict.
    """
    params, histos = {}, {}
    import os, glob, re
    re_pfname = re.compile(pfname) if pfname else None
    numruns = len(dirs)
    for num, d in enumerate(sorted(dirs)):
        run = os.path.basename(d)
        if verbosity >= 2 or (verbosity >= 1 and (num % 100 == 99 or num == 0)):
            pct = 100*(num+1)/float(numruns)
            print "Reading run '%s' data: %d/%d = %2.0f%%" % (run, num+1, numruns, pct)
        files = glob.glob(os.path.join(d, "*"))
        for f in files:
            ## Params file
            #if os.path.basename(f) == pfname:
            if re_pfname and re_pfname.search(os.path.basename(f)):
                params[run] = read_paramsfile(f)
            ## Histo file
            else:
                try:
                    ## Read as a path -> Histo dict
                    hs = read_histos(f)
                    ## Restructure into the path -> run -> Histo return dict
                    for path, hist in hs.iteritems():
                        histos.setdefault(path, {})[run] = hist
                except:
                    pass #< skip files that can't be read as histos

        ## Check that a params file was found and read in this dir... or that no attempt was made to find one
        if pfname:
            if run not in params.keys():
                raise Exception("No params file '%s' found in run dir '%s'" % (pfname, d))
        else:
            params = None
    return params, histos


def read_params(topdir, pfname="params.dat", verbosity=0):
    """
    Read interpolation anchor point data from a provided set of run directory paths.

    Returns a dict mapping run names (i.e. rundir basenames) to
    the parameter value list for each run.
    """
    params = {}
    import os, glob, re
    re_pfname = re.compile(pfname) if pfname else None
    dirs = [x for x in glob.glob(os.path.join(topdir, "*")) if os.path.isdir(x)]
    numruns = len(dirs)
    for num, d in enumerate(sorted(dirs)):
        run = os.path.basename(d)
        if verbosity >= 2 or (verbosity >= 1 and (num % 100 == 99 or num == 0)):
            pct = 100*(num+1)/float(numruns)
            print "Reading run '%s' data: %d/%d = %2.0f%%" % (run, num+1, numruns, pct)
        files = glob.glob(os.path.join(d, "*"))
        for f in files:
            ## Params file
            #if os.path.basename(f) == pfname:
            if re_pfname and re_pfname.search(os.path.basename(f)):
                params[run] = read_paramsfile(f)

        ## Check that a params file was found and read in this dir... or that no attempt was made to find one
        if pfname:
            if run not in params.keys():
                raise Exception("No params file '%s' found in run dir '%s'" % (pfname, d))
        else:
            params = None
    return params

# TODO this is much slower --- understand why!
# # http://stackoverflow.com/questions/16415156/using-sets-with-the-multiprocessing-module
# def read_rundata(dirs, pfname="params.dat", verbosity=1, nthreads=1): #, formats="yoda,root,aida,flat"):
    # """
    # Read interpolation anchor point data from a provided set of run directory paths.

    # Returns a pair of dicts, the first mapping run names (i.e. rundir basenames) to
    # the parameter value list for each run, and the second mapping observable names
    # (i.e. histogram paths) to a run -> histo dict.
    # """
    # params, histos = {}, {}
    # import os, glob, re
    # re_pfname = re.compile(pfname) if pfname else None
    # import time, multiprocessing
    # time1 = time.time()

    # from multiprocessing.managers import Manager

    # params = {}
    # histos = {}
    # manager = SyncManager()
    # manager.start()
    # histflat = manager.list()

    # # Logic:
    # #
    # #   Only directories containing the uniquely named params file are valid,
    # #   so read the params first and ignore all directories not having one
    # #   of those and then use the runs to prepare structure for multiproc dict
    # #
    # ## The job queue
    # q = multiprocessing.Queue()
    # #
    # for d in dirs:
        # run = os.path.basename(d)
        # files = glob.glob(os.path.join(d, "*"))
        # for f in files:
            # ## Params file
            # if re_pfname and re_pfname.search(os.path.basename(f)):
                # params[run]=read_paramsfile(f)
                # # histos[run]={}
                # q.put(d)



    # import sys

    # def worker(q, rflat):
        # while True:
            # if q.empty():
                # break
            # d = q.get()
            # run = os.path.basename(d)
            # files = glob.glob(os.path.join(d, "*"))
            # for f in files:
                # ## Params file
                # if re_pfname and not  re_pfname.search(os.path.basename(f)):
                    # try:
                        # ## Read as a path -> Histo dict
                        # hs = read_histos(f)
                        # temp=[]
                        # for path, hist in hs.iteritems():
                            # rflat.append([path,run,hist])

                    # except Exception, e:
                        # print e
                        # pass #< skip files that can't be read as histos


    # workers = [multiprocessing.Process(target=worker, args=(q, histflat)) for i in range(nthreads)]
    # map(lambda x:x.start(), workers)
    # map(lambda x:x.join(),  workers)
    # time2 = time.time()
    # sys.stderr.write('\rReading took %0.2fs\n\n' % ((time2-time1)))

    # for p, r, h in histflat:
        # histos.setdefault(p, {})[r] =h

    # time3 = time.time()
    # sys.stderr.write('\rData preparaion took %0.2fs\n\n' % ((time3-time2)))



    # return params, histos


def read_all_rundata(runsdir, pfname="params.dat", verbosity=1):#, nthreads=1):
    rundirs = glob.glob(os.path.join(runsdir, "*"))
    return read_rundata(rundirs, pfname, verbosity)#, nthreads)


def read_all_rundata_yaml(yamlfile):
    from professor2.utils import mkdict
    PARAMS = {}
    HISTOS = {}
    import yaml
    print "Loading YAML from %s"%yamlfile
    Y=yaml.load(open(yamlfile))
    for num, y in enumerate(Y):
        P=mkdict()
        for p in sorted(y['Params'].keys()):
            P[p] = float(y['Params'][p])
        PARAMS[num]=P
        for f in y['YodaFiles']:
            hs = read_histos(f)
            for path, hist in hs.iteritems():
                HISTOS.setdefault(path, {})[num] = hist
    return PARAMS, HISTOS


def find_maxerrs(histos):
    """
    Helper function to find the maximum error values found for each bin in the histos double-dict.

    histos is a nested dict of DataHisto objects, indexed first by histo path and then by
    run name, i.e. it is the second of the two objects returned by read_histos().

    Returns a dict of lists of floats, indexed by histo path. Each list of floats contains the
    max error size seen for each bin of the named observable, across the collection of runs
    histos.keys().

    This functions is useful for regularising chi2 etc. computation by constraining interpolated
    uncertainties to within the range seen in the run data used to create the ipols, to avoid
    blow-up of uncertainties with corresponding distortion of fit optimisation.

    TODO: asymm version?
    """
    rtn = {}
    for hn, hs in histos.iteritems():
        numbins_h = hs.next().nbins
        maxerrs_h = []
        for ib in xrange(numbins_h):
            emax = max(h.bins[ib].err for h in hs.values())
            maxerrs_h.append(emax)
        rtn[hn] = maxerrs_h
    return rtn


def read_all_rundata_h5(fname):
    """
    Helper/transition function --- read tuning data from hdf5 file and make
    Histo, params structure.
    """
    try:
        import h5py
    except ImportError:
        raise Exception("Module h5py not found --- try pip install h5py")

    with h5py.File(fname, "r") as f:
        binNames = f.get("index")[:]
        pnames = [p for p in f.get("params").attrs["names"]]
        _P =[dict(zip(pnames, p)) for p in f.get("params")] # Parameter points as dicts

    import numpy as np
    # Strip the bin numbers and use set to get list of unique histonames
    binNamesStripped = np.array([i.split("#")[0] for i in binNames])
    histoNames = sorted(list(set(binNamesStripped)))

    # dict structure for convenient access of bins by index --- histoname : [indices ...]
    lookup = { hn : np.where(binNamesStripped == hn)[0] for hn in histoNames }

    histos = {}

    with h5py.File(fname, "r") as f:
        # Read y-values and errors
        # NOTE --- first index is the bin index, the second is the run index
        # e.g. Y[33][122] is the y-value of bin 34 in run 123
        Y    = f.get("values")[:]
        E    = f.get("errors")[:]
        xmin = f.get("xmin")[:]
        xmax = f.get("xmax")[:]

        # Useable run indices, i.e. not all things NaN
        USE = np.where(~np.all(np.isnan(Y), axis=0))[0]

        # The usable parameter points
        P = [_P[u] for u in USE]

        for hname in histoNames:
            histos[hname] = {}

            # Connection between histograms and dataset indices
            bindices = lookup[hname]

            # This is an iteration over runs
            for u in USE:
                _Y     = Y[bindices][:,u]
                _E     = E[bindices][:,u]
                _xmin  = xmin[bindices]
                _xmax  = xmax[bindices]
                _bins  = [DataBin(_a,_b,_c,_d) for _a,_b,_c,_d in zip(_xmin,_xmax,_Y,_E)]
                _histo = Histo(_bins, hname)
                histos.setdefault(hname, {})[u] = _histo

    params = { u : p for u, p in zip(USE, P) }

    return params, histos


if __name__ == "__main__":
    import sys
    P,H = read_all_rundata_h5(sys.argv[1])
    from IPython import embed
    embed()
