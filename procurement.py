from functools import reduce
import itertools
import re

from combined import And, Or, Empty, Impossible, Combined
from combined import dnf
from formal_vector import FormalVector
from util import dict_msum


class Ingredients(FormalVector):
    _ZERO = "Ingredients.NONE"


class Costs(FormalVector):
    _ZERO = "Costs.FREE"


class Cost:
    cash = Costs.named("cash")
    seconds = Costs.named("seconds")


class Registry:

    def __init__(
        self,
        kind,
        procurements,
        default_procurement,
    ):
        self.kind = kind
        self.procurements = procurements
        self.default_procurement = default_procurement
        self.registry = {}

    def lookup(self, item):
        if isinstance(item, self.kind):
            (name, _, _) = item.pure()
            return self.registry.get(name, [])
        elif isinstance(item, str):
            return self.lookup(self.kind.lookup(item))
        else:
            raise LookupError(f"Cannot look up {item}")

    def register(self, procurement):
        for product_name in procurement.product_names():
            self.registry[product_name] = \
                self.registry.get(product_name, []) + [procurement]
        return self

    def from_lines(self, lines, populate=None, fuzzy=False):
        kind = self.kind
        procurements = self.procurements
        default_procurement = self.default_procurement

        IDLE = "IDLE"
        LINE_1 = "LINE_1"
        LINE_2 = "LINE_2"

        mode = IDLE
        inputs = None

        # Add an empty line to the end to trigger entry creation if there are
        # content lines going to the last provided line.
        for line in itertools.chain(lines, [""]):

            if line.strip().startswith("#"):
                continue

            try:

                line_nonempty = line.strip()
                line_empty = not line_nonempty

                # Start an entry
                if mode is IDLE and line_nonempty:
                    mode = LINE_1
                    # 5 ingredient + 2 other ingredient | attribute1=foo bar | attribute2=3
                    # 5 ingredient + 2 other ingredient | attribute1=foo bar attribute2=3
                    # 5 ingredient + 2 other ingredient
                    segments = re.split(r'\s*\|\s*', line_nonempty)

                    # Only the first segment mark `|` is important.  Others are for
                    # legibility only.  We ignore the other segment marks by
                    # re-joining the subsequent tokens.
                    if len(segments) > 1:
                        (product_raw, attributes_raw) = (segments[0], " ".join(segments[1:]))
                    else:
                        (product_raw, attributes_raw) = (segments[0], "")

                    # Parse the product.
                    product = self.kind.parse(product_raw, populate=populate, fuzzy=fuzzy)

                    # Parse the attributes.
                    #
                    # They will generally be a space-free identifier followed
                    # by an equals, then arbitrary data until another attr= or
                    # end of line.
                    # foo1=some data foo2=other foo3=8
                    #
                    # There is syntactic sugar though, and we expand that
                    # first.
                    attributes_raw = re.sub(r'(\S+):', r'procure=\1', attributes_raw)
                    keys = [
                        (m.group(1), m.span())
                        for m in re.finditer(r'([A-Za-z_][A-Za-z_0-0]*)=', attributes_raw)
                    ]
                    end_pad = [(None, (None, None))]

                    attributes = {}
                    for ((k, (_, start)), (_, (end, _))) in zip(keys, keys[1:] + end_pad):
                        attributes[k] = attributes_raw[start:end].strip()

                    # Determine the procurement in question
                    if "procure" in attributes:
                        procure_name = attributes.pop("procure")
                        procurement = self.procurements.get(
                            procure_name,
                            self.default_procurement,
                        )
                    elif "how" in attributes:
                        procure_name = attributes.pop("how")
                        procurement = self.procurements.get(
                            procure_name,
                            self.default_procurement,
                        )
                    else:
                        procurement = self.default_procurement

                # Continue an entry with an inputs line
                elif mode is LINE_1 and line_nonempty:
                    mode = LINE_2
                    inputs = self.kind.parse(line_nonempty, populate=populate, fuzzy=fuzzy)

                # Index the single-line (non-crafting) procurement
                elif line_empty and mode is LINE_1:
                    entry = procurement.create(product, **attributes)
                    self.register(entry)
                    procurement = None
                    attributes = None
                    product = None
                    inputs = None
                    mode = IDLE

                # Index the multi-line (crafting) procurement
                elif line_empty and mode is LINE_2:
                    entry = procurement.create(product, inputs=inputs, **attributes)
                    self.register(entry)
                    procurement = None
                    attributes = None
                    product = None
                    inputs = None
                    mode = IDLE

            except TypeError:
                raise

        return self


class Procurement:

    @classmethod
    def procurement_name(cls):
        return cls.__name__.lower()

    @property
    def product_names(self):
        raise NotImplementedError()

    def requirements(self, demand):
        raise NotImplementedError()

    def cannot_meet(self, demand):
        return ValueError(f"Process {self} unable to meet demand {demand}")

    def __repr__(self):
        data = self.__dict__
        no_display = ["product", "inputs"]
        kvp = " ".join(
            f"{k}={v}" for (k, v) in data.items()
            if not k.startswith("_") and k not in no_display
        )
        return f"{data['product']} | {self.procurement_name()}: {kvp}"

    def _requires(
        self,
        cost=None,
        raw=None,
        ingredients=None,
        excess=None,
    ):
        cost = cost or Costs.zero()
        raw = raw or Ingredients.zero()
        ingredients = ingredients or Ingredients.zero()
        excess = excess or Ingredients.zero()
        return {
            "cost": cost,
            "raw": raw,
            "ingredients": ingredients,
            "excess": excess,
        }


class Get(Procurement):

    @classmethod
    def create(cls, product, **kwargs):
        return cls(product, **kwargs)

    def __init__(self, product, **kwargs):
        self.product = product
        for (arg, value) in kwargs.items():
            setattr(self, arg, value)

    def product_names(self):
        (name, _, _) = self.product.pure()
        return [name]

    def requirements(self, demand=None):
        if demand is None:
            demand = self.product

        if demand == demand.zero():
            return self._requires()

        (_, count, _) = self.product.pure()
        (_, amount, _) = demand.pure()

        return self._requires(cost=0)


class Craft(Procurement):

    @classmethod
    def create(cls, product, inputs, seconds=0, **kwargs):
        if len(product.nonzero_components) > 1:
            return CraftHeterogeneous(product, inputs, seconds=seconds, **kwargs)
        else:
            return CraftHomogeneous(product, inputs, seconds=seconds, **kwargs)

    def __init__(self, product, inputs, seconds=0):
        self.product = product
        self.inputs = inputs
        self.seconds = float(seconds)


class CraftHomogeneous(Craft):

    @classmethod
    def create(cls, product, inputs, seconds=0, **kwargs):
        return cls(product, inputs, seconds=seconds, **kwargs)

    def product_names(self):
        (name, _, _) = self.product.pure()
        return [name]

    def requirements(self, demand=None):
        if demand is None:
            demand = self.product

        if demand == demand.zero():
            return self._requires()

        (name, count, _) = self.product.pure()

        demand_components = demand.nonzero_components
        if (
            len(demand_components) != 1 or
            list(demand_components) != [name]
        ):
            raise self.cannot_meet(demand)

        (_, amount, _) = demand.pure()

        if self.seconds != 0:
            rate = count / self.seconds
            return self._requires(
                ingredients=amount/rate * self.inputs,
                cost=self.seconds * Cost.seconds,
            )

        else:
            return self._requires(
                ingredients=amount/count * self.inputs,
            )


class CraftHeterogeneous(Craft):

    @classmethod
    def create(cls, product, inputs, seconds=0, **kwargs):
        return cls(product, inputs, seconds=seconds, **kwargs)

    def product_names(self):
        return list(self.product.nonzero_components.keys())

    def requirements(self, demand=None):
        if demand is None:
            demand = self.product

        if demand == demand.zero():
            return self._requires()

        demand_components = demand.nonzero_components
        product_components = self.product.nonzero_components

        if not set(demand_components).issubset(product_components):
            raise self.cannot_meet(demand)

        ratios = {
            k: demand_components[k] / product_components[k]
            for k in demand_components
        }

        # We return the minimum requirements to meet the demand.  Note that
        # this can produce excess of the product.

        bottleneck = max(ratios.values())

        if self.seconds is not None:
            reqs = bottleneck * self.seconds * self.inputs
            return self._requires(
                ingredients=reqs,
                excess=bottleneck*self.product - demand,
                cost=self.seconds * Cost.seconds,
            )

        else:
            return self._requires(
                ingredients=bottleneck * self.inputs,
                excess=bottleneck*self.product - demand,
            )


class Buy(Get):

    @classmethod
    def create(cls, product, cost, **kwargs):
        return cls(product, cost, **kwargs)

    def __init__(self, product, cost):
        self.product = product
        self.cost = float(cost)

    def requirements(self, demand=None):
        if demand is None:
            demand = self.product

        if demand == demand.zero():
            return self._requires()

        (_, count, _) = self.product.pure()
        (_, amount, _) = demand.pure()

        return self._requires(
            cost=amount/count * self.cost * Cost.cash,
        )


class Gather(Get):

    @classmethod
    def create(cls, product, seconds, **kwargs):
        return cls(product, seconds, **kwargs)

    def __init__(self, product, seconds):
        self.product = product
        self.seconds = float(seconds)

    def requirements(self, demand=None):
        if demand is None:
            demand = self.product

        if demand == demand.zero():
            return self._requires()

        (_, count, _) = self.product.pure()
        (_, amount, _) = demand.pure()

        return _requires(
            cost=amount/count * self.seconds * Cost.seconds,
        )


def _positive(vec):
    return vec.sum(
        vec.project(k) for (k, v) in vec.components.items() if v > 0
    )


def join_opt_results(results):
    # Traverse Ands looking for And "leaves"
    if isinstance(results, And):

        # If we are at an And "leaf", we can combine its children
        if all(isinstance(r, dict) for r in results):
            # Assumes the values from each result dict are monoids
            joined = dict_msum(results)
            return joined

        # Otherwise, just traverse
        else:
            return And.flat(join_opt_results(x) for x in results)

    # Traverse Ors
    elif isinstance(results, Or):
        return Or.flat(join_opt_results(x) for x in results)

    else:
        return results


def _opt_result(processes=None, raw=None, cost=None, evaluated_cost=None):
    processes = processes or []
    raw = raw or Ingredients.zero()
    cost = cost or Costs.zero()
    evaluated_cost = evaluated_cost or 0
    return {
        "processes": processes,
        "raw": raw,
        "cost": cost,
        "evaluated_cost": evaluated_cost,
    }


def _optimize_leaf(
    registry,
    cost_evaluator,
    demand,
    process,
    requirements,
    evaluated_cost,
    path=None,
):
    (name, _, _) = demand.pure()
    path = path + [name] if path else [name]
    ingredients = requirements["ingredients"]
    excess = requirements["excess"]
    cost = requirements["cost"]
    process_data = {
        "component": path,
        "process": process,
        "demand": demand,
        "ingredients": ingredients,
        "excess": excess,
        "cost": cost,
        "evaluated_cost": evaluated_cost,
    }
    base = _opt_result(
        processes=[process_data],
        cost=cost,
        evaluated_cost=evaluated_cost,
    )
    component_processes = optimize(
        registry,
        ingredients,
        cost_evaluator=cost_evaluator,
        path=path,
    )
    return join_opt_results(And.of(base, component_processes))


def optimize(registry, demand, cost_evaluator=None, path=None):
    path = path or []

    if cost_evaluator is None:
        cost_evaluator = lambda x: x["cost"].normsquare()

    if len(demand.nonzero_components) == 0:
        return _opt_result()

    elif len(demand.nonzero_components) == 1:
        options = [
            (
                process,
                reqs := process.requirements(demand),
                cost_evaluator(reqs),
            )
            for process in registry.lookup(demand)
        ]

        # FIXME: Does pure() break if there are spurious nonzero components?
        (name, _, _) = demand.pure()
        taboo = path + [name]

        candidates = [
            (process, reqs, cost) for (process, reqs, cost) in options
            if not any(c in taboo for c in reqs["ingredients"].nonzero_components)
        ]

        # If there are no ways to meet this demand, the demand itself is
        # fundamental
        if not candidates:
            return _opt_result(raw=demand)

        # Otherwise, optimize the process
        else:
            return Or.flat(
                _optimize_leaf(
                    registry=registry,
                    cost_evaluator=cost_evaluator,
                    demand=demand,
                    process=process,
                    requirements=requirements,
                    evaluated_cost=evaluated_cost,
                    path=path,
                )
                for (process, requirements, evaluated_cost)
                in candidates
            )

    # If our input is a combination of components, optimize each separately and
    # add the results
    else:
        component_processes = [
            optimize(
                registry,
                demand.project(k),
                cost_evaluator=cost_evaluator,
                path=path,
            )
            for k in demand.nonzero_components
        ]
        return join_opt_results(And.flat(component_processes))


def process_tree_overview(tree, path=None, pretty=False):
    path = path or []

    if pretty:
        pad = "    "*len(path)
        joiner = ",\n"
        start_list = "[\n"
        end_list = f"\n{pad}]"
    else:
        pad = ""
        joiner = ","
        start_list = "["
        end_list = "]"

    if isinstance(tree, Combined):
        name = tree.__class__.__name__
        s = [
            process_tree_overview(tree[i], path=path + [i], pretty=pretty)
            for (i, item) in enumerate(tree.items)
        ]
        return f"{pad}{name}{start_list}{joiner.join(s)}{end_list}"

    elif isinstance(tree, tuple) and tree[0] == "process":
        process = tree[1]["processes"][0]["process"]
        item = tree[1]["processes"][0]["component"][-1]
        return process_tree_overview(f"{process}", path=path, pretty=pretty)

    elif isinstance(tree, tuple) and tree[0] == "raw":
        parts = list(tree[1]["raw"].nonzero_components)
        if len(parts) == 0:
            raise ValueError("Wut")
        elif len(parts) == 1:
            raws = parts[0]
        else:
            raws = And.flat(parts)
        return process_tree_overview(raws, path=path, pretty=pretty)

    elif isinstance(tree, dict) and tree.get("processes") and tree.get("raw"):
        return process_tree_overview(
            And.of(("process", tree), ("raw", tree)),
            path=path,
            pretty=pretty,
        )

    elif isinstance(tree, dict) and tree.get("processes"):
        return process_tree_overview(
            ("process", tree),
            path=path,
            pretty=pretty,
        )

    elif isinstance(tree, dict) and tree.get("raw"):
        return process_tree_overview(
            ("raw", tree),
            path=path,
            pretty=pretty,
        )

    else:
        return f"{pad}{tree}"


def pprint_process_tree_overview(tree):
    print(process_tree_overview(tree, pretty=True))
