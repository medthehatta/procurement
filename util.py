import itertools
from pathlib import Path

from cytoolz import unique


def procure_alias(kind, name):
    return type(name, (kind,), {})


def paths_relative_to(path):
    """
    Return a function that converts relative to absolute paths.

    Paths will be siblings of `path`.

    E.G., inside my_module.py::

        from helpers.path_helpers import paths_relative_to

        relpath = helpers.paths_relative_to(__file__)

        # ...

        absolute_path = relpath("../assets/my-file.txt")

    """
    fullpath = Path(path).resolve()
    prefix_ = fullpath.parent

    def _relative(subpath):
        return str(prefix_ / subpath)

    return _relative


def paths_within(path):
    """
    Return a function that converts relative to absolute paths.

    Paths will be children of `path` (therefore `path` must be a directory).

    E.G., inside my_module.py::

        from helpers.path_helpers import paths_relative_to

        relpath = helpers.paths_relative_to("my_dir")

        # ...

        my_dir_assets_my_file = relpath("assets/my-file.txt")

    """
    prefix_ = Path(path).resolve()

    if not prefix_.is_dir():
        raise RuntimeError(
            f"Can only do paths within directories, but {prefix_} is not a "
            f"directory"
        )

    def _relative(subpath):
        return str(prefix_ / subpath)

    return _relative


def all_subclasses(cls):
    return list(unique(_all_subclasses(cls)))


def _all_subclasses(cls):
    if hasattr(cls, "__subclasses__"):
        for subclass in cls.__subclasses__():
            yield subclass
            yield from all_subclasses(subclass)
    else:
        return []


def dict_msum(dicts):
    result = {}
    for dic in dicts:
        for k in dic:
            if k in result:
                result[k] += dic[k]
            else:
                result[k] = dic[k]
    return result
