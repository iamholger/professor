def chi2_bins(ibins, dbins, params, errmaxs=None):
    """
    Calculate the chi2 statistic over a set of bins.

    TODO:
      * allow either direct or dict iterables
      * support a covariance matrix / nuisance correlation info argument
    """
    assert len(ibins) == len(dbins)
    chi2, ndf = 0.0, 0
    # covmatfile = ihist.path.replace("/", "-").strip("-") + ".txt"
    # covmat = np.loadtxt(covmatfile) #< error handling? else np.identity([len(ihist.bins), len(ihist.bins)])
    # TODO: Convert to double for loop or some numpy cleverness to compute the full covariance matrix chi2:
    # for nb1, (ib1, db1) in enumerate(zip(ihist.bins, dhist.bins)):
    #     for nb2, (ib2, db2) in enumerate(zip(ihist.bins, dhist.bins)): #< only loop over upper triangle?
    #chi2_bs, y_bs, ey_bs = [], [], []
    for nb, (ib, db) in enumerate(zip(ibins, dbins)):
        # print type(ib), type(db)
        ival = ib.val(params)
        #y_bs.append(ival)
        # TODO: make this observable-specific?
        ipolerr = 0.03*ival #< flat 2% extra error from ipol residuals studies, as an uncorrelated theory uncertainty
        dy = ival - db.val
        ierr = ib.err(params)
        if errmaxs:
            ierr = min(errmaxs[nb], ierr)
        #ey_bs.append(ierr)
        derr = db.err
        # if ierr > 5*derr:
        #     print "ERRS", derr, ierr
        ey2 = derr**2 + ierr**2 + ipolerr**2
        if dy and not ey2:
            print "WARNING: Zero error for non-zero bin in chi2 calculation for %s:%d. Skipping..." % (ihist, nb)
            continue
        if ey2:
            chi2_b = dy**2 / ey2
            # print chi2_b
            #chi2_bs.append(chi2_b)
            chi2 += chi2_b
            ndf += 1
    # if debug:
    #     print "bin values    =", y_bs
    #     print "bin val errs  =", ey_bs
    #     print "chi2s per bin =", chi2_bs
    return chi2, ndf


def chi2_hist(ihist, dhist, params, errmaxs=None):
    """
    Calculate the chi2 statistic over a single histogram.

    TODO:
      * support a covariance matrix / nuisance correlation info argument
    """
    assert len(ihist.bins) == len(dhist.bins)
    chi2, ndf = chi2_bins(ihist.bins, dhist.bins, params, errmaxs)
    return chi2, ndf


def chi2_hists(ihists, dhists, params, hnames=None, errmaxs={}):
    """
    Calculate the chi2 statistic over a set of histograms.

    TODO:
      * allow either direct or dict iterables
      * support a covariance matrix / nuisance correlation info argument
    """
    chi2_tot, ndf_tot = 0, 0
    if not hnames:
        hnames = ihists.keys()
    for hn in hnames:
        chi2_h, ndf_h = chi2_hist(ihists[hn], dhists[hn], params, errmaxs.get(hn))
        chi2_tot += chi2_h
        ndf_tot += ndf_h
    return chi2_tot, ndf_tot

# Alias
chi2_tot = chi2_hists
