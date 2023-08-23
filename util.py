from procurement import Craft
from procurement import Ingredients


def procure_alias(kind, name):
    return type(name, (kind,), {})


def i(n):
    return Ingredients.lookup(n)


