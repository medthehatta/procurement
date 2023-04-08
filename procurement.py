from functools import reduce
import itertools
import re

from formal_vector import FormalVector


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

        for line in lines:

            line_nonempty = line.strip()
            line_empty = not line_nonempty

            # Start an entry
            if mode is IDLE and line_nonempty:
                mode = LINE_1
                # 5 ingredient + 2 other ingredient | attribute1=foo bar | attribute2=3
                # 5 ingredient + 2 other ingredient | attribute1=foo bar attribute2=3
                # 5 ingredient + 2 other ingredient
                segments = re.split(r'\s*\|\s*', line_nonempty)

                # Only the first segment mark is important.  Others are for
                # legibility.  We ignore the other segment marks by re-joining
                # the subsequent tokens.
                if len(segments) > 1:
                    (output_raw, attributes_raw) = (segments[0], " ".join(segments[1:]))
                else:
                    (output_raw, attributes_raw) = (segments[0], "")

                # Parse the output.
                output = self.kind.parse(output_raw, populate=populate, fuzzy=fuzzy)

                # Parse the attributes.
                # They will always be a space-free identifier followed by an
                # equals, then arbitrary data until another attr= or end of
                # line.
                # foo1=some data foo2=other foo3=8
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
                else:
                    procurement = self.default_procurement

            # Continue an entry with an inputs line
            elif mode is LINE_1 and line_nonempty:
                mode = LINE_2
                inputs = self.kind.parse(line_nonempty, populate=populate, fuzzy=fuzzy)

            # Index the single-line (non-crafting) procurement
            elif line_empty and mode is LINE_1:
                entry = procurement.create(output, **attributes)
                self.register(entry)
                procurement = None
                attributes = None
                output = None
                inputs = None
                mode = IDLE

            # Index the multi-line (crafting) procurement
            elif line_empty and mode is LINE_2:
                entry = procurement.create(output, inputs=inputs, **attributes)
                self.register(entry)
                procurement = None
                attributes = None
                output = None
                inputs = None
                mode = IDLE

        return self


class Procurement:

    @property
    def product_names(self):
        raise NotImplementedError()

    def requirements(self, demand):
        raise NotImplementedError()

    def cannot_meet(self, demand):
        return ValueError(f"Process {self} unable to meet demand {demand}")

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


class Craft(Procurement):

    @classmethod
    def create(cls, output, inputs, seconds=None, **kwargs):
        if len(output.nonzero_components) > 1:
            return CraftHeterogeneous(output, inputs, seconds=seconds, **kwargs)
        else:
            return CraftHomogeneous(output, inputs, seconds=seconds, **kwargs)

    def __init__(self, output, inputs, seconds=None):
        self.output = output
        self.inputs = inputs
        self.seconds = None if seconds is None else float(seconds)


class CraftHomogeneous(Craft):

    @classmethod
    def create(cls, output, inputs, seconds=None, **kwargs):
        return cls(output, inputs, seconds=seconds, **kwargs)

    def product_names(self):
        (name, _, _) = self.output.pure()
        return [name]

    def requirements(self, demand=None):
        if demand is None:
            demand = self.output

        if demand == demand.zero():
            return self._requires()

        (name, count, _) = self.output.pure()

        demand_components = demand.nonzero_components
        if (
            len(demand_components) != 1 or
            list(demand_components) != [name]
        ):
            raise self.cannot_meet(demand)

        (_, amount, _) = demand.pure()

        if self.seconds is not None:
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
    def create(cls, output, inputs, seconds=None, **kwargs):
        return cls(output, inputs, seconds=seconds, **kwargs)

    def product_names(self):
        return list(self.output.nonzero_components.keys())

    def requirements(self, demand=None):
        if demand is None:
            demand = self.output

        if demand == demand.zero():
            return self._requires()

        demand_components = demand.nonzero_components
        output_components = self.output.nonzero_components

        if not set(demand_components).issubset(output_components):
            raise self.cannot_meet(demand)

        ratios = {
            k: demand_components[k] / output_components[k]
            for k in demand_components
        }

        # We return the minimum requirements to meet the demand.  Note that
        # this can produce excess of the output products.

        bottleneck = max(ratios.values())

        if self.seconds is not None:
            reqs = bottleneck * self.seconds * self.inputs
            return self._requires(
                ingredients=reqs,
                excess=bottleneck*self.output - demand,
                cost=self.seconds * Cost.seconds,
            )

        else:
            return self._requires(
                ingredients=bottleneck * self.inputs,
                excess=bottleneck*self.output - demand,
            )


class Buy(Procurement):

    @classmethod
    def create(cls, product, cost, **kwargs):
        return cls(product, cost, **kwargs)

    def __init__(self, product, cost):
        self.product = product
        self.cost = float(cost)

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

        return self._requires(
            cost=amount/count * self.cost * Cost.cost,
        )


class Gather(Procurement):

    @classmethod
    def create(cls, product, seconds, **kwargs):
        return cls(product, seconds, **kwargs)

    def __init__(self, product, seconds):
        self.product = product
        self.seconds = float(seconds)

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

        return _requires(
            cost=amount/count * self.seconds * Cost.seconds,
        )


def _positive(vec):
    return vec.sum(
        vec.project(k) for (k, v) in vec.components.items() if v > 0
    )


def _sum(seq):
    s = iter(seq)
    first = next(s)
    return reduce(lambda acc, x: acc + x, s, first)


def _join_opt_results(results):
    # Assumes the values of each result dict are monoids
    return {k: _sum(x[k] for x in results) for k in _opt_result()}


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


def optimize(registry, demand, cost_evaluator=None, path=None):

    if cost_evaluator is None:
        cost_evaluator = lambda x: x["cost"].normsquare()

    if len(demand.nonzero_components) == 0:
        return _opt_result()

    elif len(demand.nonzero_components) == 1:
        options = registry.lookup(demand)

        # If there are no ways to meet this demand, the demand itself is
        # fundamental
        if not options:
            return _opt_result(raw=demand)

        # Otherwise, optimize the process
        # FIXME: first pass, take the option with lowest cost
        candidates = [
            (
                process,
                reqs := process.requirements(demand),
                cost_evaluator(reqs),
            )
            for process in options
        ]
        (process, requirements, evaluated_cost) = min(
            candidates,
            key=lambda x: x[2],
        )

        # Compute the data for this process
        (name, _, _) = demand.pure()
        path = path or [name]
        ingredients = requirements["ingredients"]
        excess = requirements["excess"]
        cost = requirements["cost"]
        process_data = {
            "component": path,
            "process": type(process),
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

        # Join this to optimized processes for the components
        missing = _positive(ingredients - excess)
        component_processes = optimize(
            registry,
            missing,
            cost_evaluator=cost_evaluator,
            path=path,
        )
        return _join_opt_results([base, component_processes])

    # If our input is a combination of components, optimize each separately and
    # add the results
    else:
        path = path or []
        component_processes = [
            optimize(
                registry,
                demand.project(k),
                cost_evaluator=cost_evaluator,
                path=path+[k],
            )
            for k in demand.nonzero_components
        ]
        return _join_opt_results(component_processes)
