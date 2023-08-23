from procurement import Craft
from procurement import Buy
from procurement import Gather
from procurement import Ingredients
from procurement import Costs
from procurement import Cost
from procurement import Registry


class Vendor(Buy):

    def __init__(self, product, cash, source):
        super().__init__(product, cash)
        self.source = source


def i(n):
    return Ingredients.lookup(n)


registry = Registry(
    kind=Ingredients,
    procurements={
        "vendor": Vendor,
    },
    default_procurement=Craft,
)


with open("recipe_generic_scratch.txt") as f:
    registry.from_lines(f)
