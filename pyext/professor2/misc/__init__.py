

# TODO: move to a stats submodule, or similar
def pull(dbin, cbin, ppoint=None):
    """
    Pull between databin dbin and comparison bin (cbin).
    If ppoint is None, assume mc bins, otherwise assume ipol bin.

    TODO: "assume XXXX bins" is a nasty design: it would be better to compute the pull on the results list from the val calls
    """
    if dbin.err > 0:
        if ppoint is not None:
            return (dbin.val - cbin.val(ppoint))/dbin.err
        else:
            return (dbin.val - cbin.val)/dbin.err
    else:
        return 0
