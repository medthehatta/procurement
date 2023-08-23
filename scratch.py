import itertools
from pprint import pprint

from procurement import Craft
from procurement import CraftHomogeneous
from procurement import CraftHeterogeneous
from procurement import Buy
from procurement import Gather
from procurement import Ingredients
from procurement import Costs
from procurement import Cost
from procurement import Registry


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
    },
    default_procurement=Craft,
)


with open("recipe_generic_scratch.txt") as f:
    registry.from_lines(f)


def _positive(vec):
    return vec.sum(
        vec.project(k) for (k, v) in vec.components.items() if v > 0
    )


def optimize(registry, demand, path=None):

    if len(demand.nonzero_components) == 0:
        return {"processes": [], "raw": registry.kind.zero()}

    elif len(demand.nonzero_components) == 1:
        options = registry.lookup(demand)

        # If there are no ways to meet this demand, the demand itself is fundamental
        if not options:
            return {"processes": [], "raw": demand, "cost": Costs.zero()}

        # Otherwise, optimize the process
        # FIXME: first pass, take the option with lowest cost
        candidates = [
            (process, process.requirements(demand))
            for process in options
        ]
        (process, requirements) = min(candidates, key=lambda x: x[1]["cost"].normsquare())

        # Compute the data for this process
        (name, _, _) = demand.pure()
        path = path or [name]
        ingredients = requirements["ingredients"]
        excess = requirements["excess"]
        cost = requirements["cost"]
        base = {
            "processes": [
                {
                    "component": path,
                    "process": type(process),
                    "demand": demand,
                    "ingredients": ingredients,
                    "excess": excess,
                    "cost": cost,
                },
            ],
            "raw": registry.kind.zero(),
            "cost": cost,
        }

        # Join this to optimized processes for the components
        missing = _positive(ingredients - excess)
        component_processes = optimize(registry, missing, path=path)
        return {
            "processes": base["processes"] + component_processes["processes"],
            "raw": registry.kind.sum(x["raw"] for x in [base, component_processes]),
            "cost": base["cost"] + component_processes["cost"],
        }

    # If our input is a combination of components, optimize each separately and
    # add the results
    else:
        path = path or []
        component_processes = [
            optimize(registry, demand.project(k), path=path+[k])
            for k in demand.nonzero_components
        ]
        return {
            "processes": list(
                itertools.chain.from_iterable(x["processes"] for x in component_processes)
            ),
            "raw": registry.kind.sum(x["raw"] for x in component_processes),
            "cost": Costs.sum(x["cost"] for x in component_processes),
        }
