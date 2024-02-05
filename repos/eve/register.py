from procurement import Procurement
from procurement import Buy
from procurement import Cost
from procurement import Costs
from procurement import Craft
from procurement import CraftHeterogeneous
from procurement import CraftHomogeneous
from procurement import Gather
from procurement import Ingredients
from procurement import Registry

from util import all_subclasses
from util import procure_alias
from util import paths_relative_to
from util import dict_msum


relpath = paths_relative_to(__file__)


class Manufacturing(CraftHomogeneous):

    def __init__(self, output, inputs, seconds=None, tax=0, mat_reduction_pct=0):
        super().__init__(output, inputs, seconds)
        self.tax = float(tax)
        self.mat_reduction_pct = float(mat_reduction_pct)

    def requirements(self, demand=None):
        required1 = super().requirements(demand)
        mat_ratio = (100 - self.mat_reduction_pct)/100
        required1["ingredients"] = mat_ratio * required1["ingredients"]
        # TODO: Actually add tax
        return required1


PlanetaryCraft = procure_alias(Manufacturing, "PlanetaryCraft")

Fitting = procure_alias(Manufacturing, "Fitting")


class SystemBuy(Buy):

    def __init__(self, product, cost, where=None):
        super().__init__(product, cost)
        self.where = where

    @classmethod
    def procurement_name(cls):
        return "buy"


registry = Registry(
    kind=Ingredients,
    default_procurement=Manufacturing,
    procurements={
        "planetary": PlanetaryCraft,
        **{
            c.procurement_name(): c for c in all_subclasses(Procurement)            
        },
        # Override Buy in EVE to be SystemBuy
        "buy": SystemBuy,
        "fit": Fitting,
    },
)


with open(relpath("recipes.txt")) as f:
    registry.from_lines(f)
