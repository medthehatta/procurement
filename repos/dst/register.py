
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


registry = Registry(
    kind=Ingredients,
    default_procurement=Craft,
    procurements={
        c.procurement_name(): c for c in all_subclasses(Procurement)
    },
)


with open(relpath("recipes.txt")) as f:
    registry.from_lines(f)
