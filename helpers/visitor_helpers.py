def handles(*kinds):
    """Decorator that registers a visitor handler method for the given pyslang node kinds."""
    def decorator(fn):
        fn._handles = kinds
        return fn
    return decorator


def build_lookup_table(visitor):
    """Build a {kind: handler_method} dispatch table from all @handles-decorated methods."""
    table = {}
    for name in dir(type(visitor)):
        method = getattr(type(visitor), name, None)
        if method is None:
            continue
        for kind in getattr(method, '_handles', ()):
            table[kind] = getattr(visitor, name)
    return table
