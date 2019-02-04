#/usr/bin/env python

import numpy as np

import sys

newstuff=sys.argv[1]
goftune = float(sys.argv[2])
outfile=sys.argv[3]


def mkGoFPlot(Xnew, Xtune, outputfile):
    import numpy as np
    X=np.sqrt(2)*np.loadtxt(Xnew)
    g_68 = np.percentile(X, 68.8)
    g_95 = np.percentile(X, 95)
    g_99 = np.percentile(X, 99.9)


    import pylab
    pylab.axvspan(min(X), g_68, label="68.8 pct", facecolor="b", alpha=0.1)
    pylab.axvspan(g_68, g_95, label="95 pct", facecolor="r", alpha=0.1)
    # pylab.axvspan(g_95, g_99, label="99.9 pct", facecolor="g", alpha=0.1)
    pylab.hist(X, "auto", normed=True, histtype="step")
    pylab.axvline(Xtune, label="Tune")
    pylab.legend()
    pylab.xlabel(r"$\phi^2$")
    pylab.ylabel(r"$p(X = \phi^2)$")
    pylab.savefig(outputfile)


mkGoFPlot(newstuff, goftune, outfile)
