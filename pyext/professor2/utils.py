# -*- python -*-

def mkdict():
    """Return an OrderedDict if possible, or a normal dict if not."""
    try:
        from collections import OrderedDict
        return OrderedDict()
    except:
        try:
            from ordereddict import OrderedDict
            return OrderedDict()
        except:
            return dict()
