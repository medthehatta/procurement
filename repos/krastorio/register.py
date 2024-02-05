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


relpath = paths_relative_to(__file__)


x = procure_alias(CraftHomogeneous, "x")

BurnerMiner = procure_alias(Gather, "BurnerMiner")

StoneFurnace = procure_alias(CraftHomogeneous, "StoneFurnace")

Character = procure_alias(CraftHomogeneous, "Character")
BasicAssembler = procure_alias(CraftHomogeneous, "BasicAssembler")


procurements = {
    c.procurement_name(): c for c in all_subclasses(Procurement)
}
registry = Registry(
    kind=Ingredients,
    default_procurement=BasicAssembler,
    procurements=procurements,
)


with open(relpath("recipes.txt")) as f:
    registry.from_lines(f)
