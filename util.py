from procurement import Craft
from procurement import Ingredients
from procurement import Registry


def procure_alias(kind, name):
    return type(name, (kind,), {})


def i(n):
    return Ingredients.lookup(n)


def ip(n, **kwargs):
    return Ingredients.parse(n, **kwargs)


def default_registry(procurements):
    return Registry(
        kind=Ingredients,
        default_procurement=Craft,
        procurements=procurements,
    )
