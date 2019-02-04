
def exploreValley(center_t, direction_t, TRAFO, GOFdef,):
    exec GOFdef in globals() # Note globals!
    def getVal(a):
        temp_t = center_t +  a*direction_t
        temp = TRAFO * temp_t.transpose()
        temp_r = temp.transpose().tolist()[0]
        return profGoF(*temp_r) - target


def mkNewParamCorrelation(T_trans, T, point, GOFdef, target):
    import professor2 as prof
    exec GOFdef in locals() # Note globals!

    from numpy import matrix
    rv = matrix(point.values())
    center_t = (T_trans * rv.transpose()).transpose()

    DIM = len(point.values())
    from scipy.optimize import fsolve
    from numpy import zeros,diag

    def getVal(a, direction):
        temp_t = center_t +  a*direction
        temp = T * temp_t.transpose()
        temp_r = temp.transpose().tolist()[0]
        return profGoF(*temp_r) - target

    def getX(a, direction):
        return center_t +  a*direction

    newdiag = []
    for i in xrange(DIM):
        ev = zeros(DIM)
        ev[i] = 1
        temp  = fsolve(lambda x:getVal(x, ev), 1)
        temp2 = fsolve(lambda x:getVal(x, -ev), 1)
        if temp > temp2:
            newdiag.append(temp)
        else:
            newdiag.append(temp2)

    N = diag([x[0] for x in newdiag])
    return T_trans*N*T




def calcHistoCov(h, COV_P, result):
    """
    Propagate the parameter covariance onto the histogram covariance
    using the ipol gradients.
    """
    IBINS = h.bins
    from numpy import zeros
    COV_H = zeros((h.nbins, h.nbins))
    from numpy import array
    for i in xrange(len(IBINS)):
        GRD_i = array(IBINS[i].grad(result))
        for j in xrange(len(IBINS)):
            GRD_j = array(IBINS[j].grad(result))
            pc =GRD_i.dot(COV_P).dot(GRD_j)
            COV_H[i][j] = pc
    return COV_H


def mkErrorPropagationHistos(IHISTOS, point, COV, combine=False):
    covs = {}
    properrs = {}
    ipolerrs = {}
    ipolvals = {}
    from numpy import sqrt
    for k, v in IHISTOS.iteritems():
        covs[k]     = calcHistoCov(v, COV, point)
        properrs[k] = sqrt(0.5*covs[k].diagonal())
        ipolerrs[k] = [b.err  for b in v.toDataHisto(point).bins]
        ipolvals[k] = [b.val  for b in v.toDataHisto(point).bins]

    scatters=[]
    for k, v in IHISTOS.iteritems():
        T = v.toDataHisto(point)
        for i in xrange(T.nbins):
            T.bins[i].errs=properrs[k][i] if combine is False else sqrt(properrs[k][i]**2 + ipolerrs[k][i]**2)
        scatters.append(T.toScatter2D())
    return scatters


def mkScatters(masterbox, mastercenter, ppoint):
    import professor2 as prof
    boxdict=None
    try:
        ppoint = ppoint.values()
    except:
        pass
    for box, bdict in masterbox.iteritems():
        if prof.pInBOX(ppoint, box):
            boxdict = bdict
            break
    if boxdict is None:
        distances={}
        for c in mastercenter.keys():
            distances[prof.pBoxDistance(ppoint, c)] = c
        winner = min(distances.keys())
        boxdict = mastercenter[distances[winner]]
    ipolH=boxdict["IHISTOS"]

    scatters_e =[]
    for k in sorted(ipolH.keys()):
        v = ipolH[k]
        T = v.toDataHisto(ppoint)
        scatters_e.append(T.toScatter2D())
    return scatters_e


def mkEnvelopes(central, etunes):
    ret = {}
    for i in xrange(1, len(etunes.keys())/2 + 1):
        ret[i] = []
        Hplus  = etunes[i]
        Hminus = etunes[-i]
        for num_h, h in enumerate(central):
            temp = h.clone()
            for num_p, p in enumerate(temp.points):
                yplus  = Hplus[num_h].points[num_p].y
                yminus = Hminus[num_h].points[num_p].y
                if yplus > p.y:
                    eplus  = yplus - p.y
                    eminus = p.y - yminus
                else:
                    eplus  = yminus - p.y
                    eminus  = p.y - yplus
                p.yErrs = (eminus, eplus)
            ret[i].append(temp)
    return ret


def mkTotvelopes(central, etunes):
    ret = []
    for num_h, h in enumerate(central):
        temp = h.clone()
        allThis = [x[num_h] for x in etunes.values()]
        for num_p, p in enumerate(temp.points):
            dybin = [x.points[num_p].y - p.y for x in allThis]
            pos = [x for x in dybin if x>=0]
            neg = [x for x in dybin if x<0]
            eplus = max(pos) if len(pos) > 0 else 0
            eminus = abs(min(neg)) if len(neg) > 0 else 0
            p.yErrs = (eminus, eplus)
        ret.append(temp)
    return ret


def mkAddvelopes(central, etunes, addLinear=False):
    ret = []
    for num_h, h in enumerate(central):
        temp = h.clone()
        allThis = [x[num_h] for x in etunes.values()]
        for num_p, p in enumerate(temp.points):
            dybin = [x.points[num_p].y-p.y for x in allThis]
            pos = [x for x in dybin if x>=0]
            neg = [x for x in dybin if x<0]
            from math import sqrt
            if addLinear:
                eplus  = sum(pos)                   if len(pos) > 0 else 0
                eminus = sum([abs(x) for x in neg]) if len(neg) > 0 else 0
            else:
                eplus  = sqrt(sum([x*x for x in pos])) if len(pos) > 0 else 0
                eminus = sqrt(sum([x*x for x in neg])) if len(neg) > 0 else 0
            p.yErrs = (eminus, eplus)
        ret.append(temp)
    return ret


def readFromFile(fname):
    nbins = 0
    params = []
    ret = []
    with open(fname) as f:
        for line in f:
            l = line.strip()
            if l.startswith("#"):
                if "Nbins" in l:
                    nbins = int(l.split()[-1])
                elif "Parameters" in l:
                    params = l.split()[2:]
            else:
                ret.append(map(float, l.split()))
    return ret, nbins, params

# https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
def mkTabulate(head, rows):
    from tabulate import tabulate
    return tabulate(rows, headers=head,  tablefmt='orgtbl')

def mkTable(head, rows):
    t = head[0]
    for h in head[1:]:
        t+="\t%s"%str(h)

    for row in rows:
        t+="\n%s"%row[0]
        for r in row[1:]:
            t+="\t%s"%str(r)
    t+="\n"

    return t

def mkEigenTable(ETs, rfile, latex=False):
    import professor2 as prof
    P_min, OTH = prof.readResult(rfile)

    head = ["Tune"] + P_min.keys()

    rows = [["Central"] + P_min.values()]

    for e in sorted(list(set([abs(x) for x in ETs.keys()]))):
        rows.append(["%i+"%e] + list(ETs[e]) )
        rows.append(["%i-"%e] + list(ETs[-e]))

    s="\nEigentune summary:\n\n"

    try:
        from tabulate import tabulate
        s+=mkTabulate(head, rows)
    except ImportError, e:
        print e
        print "Fallback to simple table"
        s+=mkTable(head, rows)
    return s

