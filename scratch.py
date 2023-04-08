import itertools
from pprint import pprint

from combined import And, Or, Empty, Impossible
from combined import dnf

from procurement import Craft
from procurement import CraftHomogeneous
from procurement import CraftHeterogeneous
from procurement import Buy
from procurement import Gather
from procurement import Ingredients
from procurement import Costs
from procurement import Cost
from procurement import Registry
from procurement import optimize
from procurement import join_opt_results


class Vendor(Buy):

    def __init__(self, product, cost, source):
        super().__init__(product, cost)
        self.source = source


def procure_alias(kind, name):
    return type(name, (kind,), {})


def i(n):
    return Ingredients.lookup(n)


registry = Registry(
    kind=Ingredients,
    procurements={
        "vendor": Vendor,
        "coal liquefaction": procure_alias(CraftHeterogeneous, "CoalLiquefaction"),
        "advanced oil refining": procure_alias(CraftHeterogeneous, "AdvancedOilRefining"),
        "chemical": procure_alias(CraftHomogeneous, "Chemical"),
        "heavy cracking": procure_alias(CraftHomogeneous, "HeavyOilCracking"),
        "light cracking": procure_alias(CraftHomogeneous, "LightOilCracking"),
    },
    default_procurement=Craft,
)


with open("recipe_generic_scratch.txt") as f:
    registry.from_lines(f)


