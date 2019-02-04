# -*- python -*-



# http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
def numCombs(n, k):
    """
    n choose k algorithm
    """
    from operator import mul
    from fractions import Fraction
    return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )

def xrandomUniqueCombinations(items, nchoose, howmany=None):
    """ Generator-like function for n choose k items """
    seencombs = []
    # Max number safeguard against infinite loops
    maxnum = numCombs(len(items), nchoose)
    import random
    if howmany is None or howmany > maxnum:
        print "Only %i possible combinations"%maxnum
        howmany = maxnum
    while len(seencombs) < howmany:
        temp = random.sample(items, nchoose)
        temp.sort()
        if not sorted(temp) in seencombs:
            seencombs.append(temp)
            yield temp


## Define a sampler type
# This is obsolete now
class Sampler(object):
    # @deprecated
    def __init__(self, low, high, bias=None):
        self.low = float(low)
        self.high = float(high)
        self.f, self.invf = None, None
        if bias is not None:
            ## Import clever machinery
            try:
                import sympy as sp
                from sympy.abc import x, y
                import numpy as np
            except ImportError:
                print "Bias functions require SymPy and NumPy to be installed... exiting"
                exit(1) #< TODO: don't exit from inside a lib function...
            ## Make transformation and its inverse
            try:
                #print bias
                f_expr = sp.sympify(bias)
            except sp.SympifyError, e:
                print "Bias function could not be parsed by SymPy:"
                print e
                exit(1) #< TODO: don't exit from inside a lib function...
            try:
                finv_exprs = sp.solve(sp.Eq(y, f_expr), x)
                finv_expr = finv_exprs[0]
                #print f_expr, finv_exprs
                self.f = sp.lambdify(x, f_expr, "numpy")
                self.finv = sp.lambdify(y, finv_expr, "numpy")
                self.lowf, self.highf = self.f(self.low), self.f(self.high)
            except Exception, e:
                print "Bias function could not be used/inverted by SymPy:"
                print e
                exit(1) #< TODO: don't exit from inside a lib function...

    def shoot(self):
        import random
        if not self.f:
            ## Just uniform sampling between low..high
            val = random.uniform(self.low, self.high)
        else:
            ## Uniform sample in transformed space, and transform the result back
            valf = random.uniform(self.lowf, self.highf)
            val = self.finv(valf)
        return val

    def __call__(self):
        return self.shoot()

    def __repr__(self):
        return "<%s with x in %f ... %f>" % (self.__class__.__name__, self.low, self.high)



class Sobol(object):

    def __init__(self, dim, initialSeed=0):
        self._seed=initialSeed
        self._dim=dim

    def shoot(self):
        try:
            import sobol
        except ImportError:
            raise Exception("sobol not available, try pip install sobol")
        p, newseed = sobol.sobol_seq.i4_sobol(self._dim, self._seed)
            # from IPython import embed
            # embed()
        self._seed=newseed
        return p

    def __call__(self):
        return self.shoot()

class LatinBox(object):

    def __init__(self, dim, initialSeed=0):
        self._seed=initialSeed
        self._dim=dim

    def shoot(self):
        try:
            import pyDOE
        except ImportError:
            raise Exception("pyDOE not available, try pip install pyDOE")
        p = pyDOE.lhs(self._dim, 1)
        return p[0]

    def __call__(self):
        return self.shoot()

class RandomU(object):

    def __init__(self, dim, seed=0):
        self._dim=dim
        from numpy import random
        random.seed(seed)

    def shoot(self):
        from numpy import random
        return random.uniform(0,1,(self._dim,1)).flatten()

    def __call__(self):
        return self.shoot()

class NDSampler(object):

    def __init__(self, ranges, biases=None, sampler="uniform", seed=0):
        try:
            from collections import OrderedDict
        except:
            from ordereddict import OrderedDict
        self._ranges = ranges
        self._seed=seed
        self._dim = len(self._ranges)
        self._sampler=sampler
        self._biases=biases
        self.f, self.invf = [None for i in xrange(self._dim)], [None for i in xrange(self._dim)]


        if type(self._biases)==list:
            if any([x is not None for x in self._biases]):
                self.setBiases()
        else:
            if self._biases is not None:
                self.setBiases()

        if sampler=="sobol":
            self._generator=Sobol(self._dim, self._seed)
        elif sampler=="uniform":
            self._generator=RandomU(self._dim, self._seed)
        elif sampler=="latin":
            self._generator=LatinBox(self._dim) # NOTE: no seed in pyDOE lhs
        else:
            raise Exception("Unknown sampling method %s not implemented"%sampler)

    def setBiases(self):
        try:
            import sympy as sp
            from sympy.abc import x, y
            import numpy as np
        except ImportError:
            raise Exception("Bias functions require numpy and sympy, try pip install sympy numpy")

        for num, b in enumerate(self._biases):
            if b is not None:
                ## Make transformation and its inverse
                try:
                    f_expr = sp.sympify(b)
                except sp.SympifyError, e:
                    raise Exception("Bias function could not be parsed by SymPy, try pip install sympy")
                try:
                    finv_exprs = sp.solve(sp.Eq(y, f_expr), x)
                    finv_expr = finv_exprs[0]
                    self.f[num] = sp.lambdify(x, f_expr, "numpy")
                    self.invf[num] = sp.lambdify(y, finv_expr, "numpy")
                except Exception, e:
                    print e
                    raise Exception("Bias function could not be used/inverted by SymPy")

    def scale(self, Praw):
        P=[]
        for num, p in enumerate(Praw):
            a, b = self._ranges[num]
            if self.f[num] is not None:
                a=self.f[num](a)
                b=self.f[num](b)
            pscaled = a + p*(b-a)
            if self.invf[num] is not None:
                pscaled = self.invf[num](pscaled)
            P.append(pscaled)
        return P


    def __call__(self):

        tryAgain=True
        while tryAgain:
            P_raw = self._generator.shoot()
            ret = self.scale(P_raw)
            for num, p in enumerate(ret):
                # Try again
                if ret[num] < self._ranges[num][0] or ret[num] > self._ranges[num][1]:
                    print "Error:", ret[num], "not in", self._ranges[num], "will try again, maybe bias is not working"
                    continue
                tryAgain=False

        return ret

    def __repr__(self):
        s="<%s,  %i-D %s>"%(self.__class__.__name__, self._dim, self._sampler)
        return s


## Test biased sampler machinery if run as main
if __name__ == "__main__":
    # This one is obsolete
    s = Sampler(1, 10, "exp(x)")
    import yoda
    h = yoda.Histo1D(20, 0, 10)
    for _ in xrange(10000):
        h.fill( s() )
    yoda.plot(h, "samplingtest_obsolete_exp_bias.pdf")


    NS = NDSampler([[1,10]], ["exp(x)"], sampler="sobol")
    h = yoda.Histo1D(20, 0, 10)
    for _ in xrange(10000):
        h.fill( NS()[0] )
    yoda.plot(h, "samplingtest_exp_bias.pdf")


    NP=500

    import pylab
    s=Sobol(2)
    PS = [s() for _ in xrange(NP)]
    X=[p[0] for p in PS]
    Y=[p[1] for p in PS]
    pylab.clf()
    pylab.plot(X,Y, "bo", label="Sobol")

    s=RandomU(2)
    PU = [s() for _ in xrange(NP)]
    X=[p[0] for p in PU]
    Y=[p[1] for p in PU]
    pylab.plot(X, Y, "rx", label="Random uniform")

    s=LatinBox(2)
    PL = [s() for _ in xrange(NP)]
    X=[p[0] for p in PL]
    Y=[p[1] for p in PL]
    pylab.plot(X, Y, "g+", label="Latin hypercube")
    pylab.legend()
    pylab.savefig("samplingtest_uni_v_sob_v_latin.pdf")


    pylab.clf()
    NS = NDSampler([[1,10], [2,3]], ["exp(x)", "10**x"], "sobol")
    P2S = [NS() for _ in xrange(NP)]
    X=[p[0] for p in P2S]
    Y=[p[1] for p in P2S]
    pylab.plot(X, Y, "bo", label="Sobol")

    NL = NDSampler([[1,10], [2,3]], ["exp(x)", "10**x"], "latin")
    P2L = [NL() for _ in xrange(NP)]
    X=[p[0] for p in P2L]
    Y=[p[1] for p in P2L]
    pylab.plot(X, Y, "g+", label="Latin HC")

    NU = NDSampler([[1,10], [2,3]], ["exp(x)", "10**x"], "uniform")
    P2U = [NU() for _ in xrange(NP)]
    X=[p[0] for p in P2U]
    Y=[p[1] for p in P2U]
    pylab.plot(X, Y, "rx", label="Random uniform")

    pylab.legend()
    pylab.savefig("samplingtest_2D_sampling_wbias.pdf")

