import itertools
from pprint import pprint

from combined import And, Or, Empty, Impossible
from combined import dnf

from procurement import Buy
from procurement import Cost
from procurement import Costs
from procurement import Craft
from procurement import CraftHeterogeneous
from procurement import CraftHomogeneous
from procurement import Gather
from procurement import Ingredients
from procurement import Registry
from procurement import join_opt_results
from procurement import optimize

from util import default_registry
from util import i
from util import procure_alias


class Vendor(Buy):

    def __init__(self, product, cost, source):
        super().__init__(product, cost)
        self.source = source


registry = default_registry(
    procurements={
        "vendor": Vendor,
        "coal liquefaction": procure_alias(CraftHeterogeneous, "CoalLiquefaction"),
        "advanced oil refining": procure_alias(CraftHeterogeneous, "AdvancedOilRefining"),
        "chemical": procure_alias(CraftHomogeneous, "Chemical"),
        "heavy cracking": procure_alias(CraftHomogeneous, "HeavyOilCracking"),
        "light cracking": procure_alias(CraftHomogeneous, "LightOilCracking"),
    },
)


with open("recipe_generic_scratch.txt") as f:
    registry.from_lines(f)


