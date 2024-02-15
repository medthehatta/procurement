from functools import reduce
from itertools import chain
import json
from uuid import uuid1
import re

from cytoolz import unique
from cytoolz import get
from cytoolz import curry
from cytoolz import groupby
import numpy as np
from scipy.optimize import milp
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds


class InteractiveRegistryExplorer:

    NO_SELECTION = "__NO_SELECTION__"

    def __init__(self, registry):
        self.registry = registry

    def interactive_pick(self, options, auto_if_solo=False):
        enumerated = list(enumerate(options, start=1))

        indexed = {str(i): opt for (i, opt) in enumerated}

        if auto_if_solo and len(options) == 0:
            return self.NO_SELECTION
        elif auto_if_solo and len(options) == 1:
            return options[0]

        while True:
            print("Make a selection:")

            for (i, opt) in enumerated:
                print(f"{i}) {opt}")

            choice = input("Choice (S to skip, empty for 1): ").strip()

            if not choice:
                choice = "1"

            if choice.strip().lower() == "s":
                return self.NO_SELECTION

            if choice in indexed:
                return indexed[choice]
            else:
                print("Invalid choice, try again.\n")

    def interactive_build_edgelist(
        self,
        desired,
        parent=None,
        path=None,
        auto_if_solo=True,
        no_recurse_part=None,
        no_recurse_path=None,
        predicate=None,
    ):
        path = path or []
        no_recurse_part = no_recurse_part or []
        no_recurse_path = no_recurse_path or []
        predicate = predicate or (lambda x: True)

        v_desired = desired.triples()
        if len(v_desired) != 1:
            raise NotImplementedError(
                f"Only supports single output atm. {desired}"
            )

        (part, _, _) = v_desired[0]
        if part in no_recurse_part:
            return None

        path = path + [part]
        if path in no_recurse_path:
            return None

        printable_path = ' > '.join(path)
        print(printable_path)
        choice = self.interactive_pick(
            self.registry.search(predicate, part=part),
            auto_if_solo=auto_if_solo,
        )
        print("")
        if choice is self.NO_SELECTION:
            return None
        else:
            instance = choice.instance(printable_path)
            yield (part, instance, parent)
            for inp in instance.inputs.nonzero_components:
                yield from self.interactive_build_edgelist(
                    instance.inputs.project(inp),
                    parent=instance,
                    path=path,
                    auto_if_solo=auto_if_solo,
                    no_recurse_part=no_recurse_part,
                    no_recurse_path=no_recurse_path,
                    predicate=predicate,
                )


class Predicates:

    @classmethod
    @curry
    def and_(cls, pred1, pred2, process):
        return pred1(process) and pred2(process)

    @classmethod
    @curry
    def or_(cls, pred1, pred2, process):
        return pred1(process) or pred2(process)

    @classmethod
    @curry
    def not_(cls, predicate, process):
        return not predicate(process)

    @classmethod
    @curry
    def outputs_part(cls, part, process):
        return part in process.output_rate.nonzero_components

    @classmethod
    @curry
    def requires_part(cls, part, process):
        return part in process.input_rate.nonzero_components

    @classmethod
    @curry
    def does_not_cost(cls, part, process):
        return part not in process.cost_rate.nonzero_components

    @classmethod
    def costs(cls, part, process):
        return part in process.cost_rate.nonzero_components

    @classmethod
    def non_character(cls, process):
        return process.process != "character"

    @classmethod
    @curry
    def furnaces_are(cls, which, process):
        if "furnace" not in process.process:
            return True
        else:
            return which in process.process


class ProcessRegistry:

    def __init__(self, factory):
        self.factory = factory
        self._registry = []
        self._output_index = {}
        self._term_index = {}

    def register(self, process):
        self._registry.append(process)
        index = len(self._registry) - 1
        for part in process.outputs.components:
            self._output_index[part] = (
                self._output_index.get(part, []) + [index]
            )
            for term in self._terms(part):
                self._term_index[term.lower()] = (
                    self._term_index.get(term.lower(), []) + [index]
                )

    def _terms(self, s):
        return s.lower().replace("-", " ").split()

    def only(self, part):
        candidates = self.find(part)
        if len(candidates) == 0:
            raise ValueError(f"No matches for '{part}'")
        elif len(candidates) == 1:
            return candidates[0]
        else:
            raise ValueError(f"Ambiguous matches for '{part}': {candidates}")

    def search(self, predicate, part=None):
        if part is not None:
            return [p for p in self.find(part) if predicate(p)]
        else:
            return [p for p in self._registry if predicate(p)]

    def find(self, part):
        return [
            self._registry[i]
            for i in self._output_index.get(part, [])
        ]

    def first(self, part):
        candidates = self.find(part)
        if len(candidates) == 0:
            raise ValueError(f"No matches for '{part}'")
        else:
            return candidates[0]

    def fuzzy(self, terms):
        return [
            self._registry[i]
            for i in reduce(
                set.intersection,
                [
                    set(self._term_index.get(term.lower(), []))
                    for term in terms.split()
                ],
            )
        ]

    def filter(self, predicate):
        return [p for p in self._registry if predicate(p)]

    def register_from_lines(self, lines):
        for p in self.factory.processes_from_lines(lines):
            self.register(p)
        return self


class ProcessFactory:

    def __init__(self, kind, process_kind, default_process):
        self.kind = kind
        self.process_kind = process_kind
        self.default_process = default_process

    def specs_from_lines(self, lines):
        found = False
        buf = ""

        for line in lines:

            if not line.strip() or line.strip().startswith("#"):
                if found:
                    yield parse_process(buf)
                    buf = ""
                    found = False

            else:
                buf += line + "\n"
                found = True

    def processes_from_lines(self, lines):
        return (
            self.process_kind.from_dict(
                self.kind,
                entry,
                default_process=self.default_process,
            )
            for entry in self.specs_from_lines(lines)
        )


class Process:

    @classmethod
    def from_dict(cls, kind, dic, default_process="process"):
        reserved = [
            "outputs",
            "inputs",
            "cost",
            "seconds",
        ]
        return cls(
            kind.parse(dic["outputs"]),
            inputs=kind.parse(dic["inputs"]) if "inputs" in dic else None,
            cost=kind.parse(dic["cost"]) if "cost" in dic else None,
            seconds=float(dic.get("seconds", 1)),
            process=dic.get("process", default_process),
            metadata={k: dic[k] for k in dic if k not in reserved},
        )

    @classmethod
    def from_transfer(cls, transfer, **kwargs):
        outputs = transfer.from_triples(
            (n, v, b) for (n, v, b) in transfer.triples()
            if v > 0
        )
        inputs = -transfer.from_triples(
            (n, v, b) for (n, v, b) in transfer.triples()
            if v < 0
        )
        return cls(outputs, inputs=inputs, **kwargs)

    def __init__(
        self,
        outputs,
        inputs=None,
        cost=None,
        seconds=1,
        process=None,
        instance_id=None,
        metadata=None,
    ):
        self.process = process or "produce"
        self.outputs = outputs
        self.inputs = inputs or type(self.outputs).zero()
        self.transfer = self.outputs - self.inputs
        self.instance_id = instance_id
        self.metadata = metadata or {}

        in_kind = type(self.inputs)
        out_kind = type(self.outputs)

        if not (
            isinstance(self.inputs, out_kind) and
            isinstance(self.outputs, in_kind)
        ):
            raise ValueError(
                f"Inputs and outputs do not have compatible kinds.  "
                f"They have {in_kind} and {out_kind} "
                f"respectively."
            )

        self.kind = out_kind
        self.cost = cost or self.kind.zero()
        self.seconds = seconds

        self.cost_rate = (1/self.seconds) * self.cost
        self.output_rate = (1/self.seconds) * self.outputs
        self.input_rate = (1/self.seconds) * self.inputs

        self.transfer_rate = self.output_rate - self.input_rate

    def __repr__(self):
        if self.instance_id:
            return (
                f"<{self.process} ({self.instance_id}) ({self.output_rate})>"
            )
        else:
            return (
                f"<{self.process} ({self.output_rate})>"
            )

    def is_free(self):
        return self.cost == self.kind.zero()

    def instance(self, instance_id=None):
        return type(self)(
            outputs=self.outputs,
            inputs=self.inputs,
            cost=self.cost,
            seconds=self.seconds,
            process=self.process,
            instance_id=instance_id or uuid1().hex,
            metadata=self.metadata,
        )


def parse_process(s):
    stripped_lines = (line.strip() for line in s.splitlines())
    lines = [
        line for line in stripped_lines
        if line and not line.startswith("#")
    ]

    if len(lines) == 0:
        raise ValueError(f"No substantive lines in (next line):\n{s}")
    elif len(lines) == 1:
        return _parse_process_header(lines[0])
    elif len(lines) == 2:
        return {
            **_parse_process_header(lines[0]),
            "inputs": lines[1],
        }
    else:
        raise ValueError(f"Found too many lines in (next line):\n{s}")


def _parse_process_header(s):
    # 5 ingredient + 2 other ingredient | attribute1=foo bar | attribute2=3
    # 5 ingredient + 2 other ingredient | attribute1=foo bar attribute2=3
    # 5 ingredient + 2 other ingredient
    segments = re.split(r'\s*\|\s*', s)

    # Only the first segment mark `|` is important.  Others are for
    # legibility only.  We ignore the other segment marks by
    # re-joining the subsequent tokens.
    if len(segments) > 1:
        (product_raw, attributes_raw) = (segments[0], " ".join(segments[1:]))
    else:
        (product_raw, attributes_raw) = (segments[0], "")

    # Parse the attributes.
    #
    # They will generally be a space-free identifier followed
    # by an equals, then arbitrary data until another attr= or
    # end of line.
    # foo1=some data foo2=other foo3=8
    #
    # There is syntactic sugar though, and we expand that
    # first.
    attributes_raw = re.sub(r'^\s*(.+):', r'process=\1', attributes_raw)
    keys = [
        (m.group(1), m.span())
        for m in re.finditer(r'([A-Za-z_][A-Za-z_0-0]*)=', attributes_raw)
    ]
    end_pad = [(None, (None, None))]

    attributes = {}
    for ((k, (_, start)), (_, (end, _))) in zip(keys, keys[1:] + end_pad):
        # Try to interpret each attribute as a valid JSON primitive; otherwise
        # take the literal string
        try:
            attributes[k] = json.loads(attributes_raw[start:end].strip())
        except json.decoder.JSONDecodeError:
            attributes[k] = attributes_raw[start:end].strip()

    return {
        "outputs": product_raw,
        **attributes,
    }


def _densify(dict_lst):
    keys = list(
        unique(
            chain.from_iterable(dic.keys() for dic in dict_lst)
        )
    )
    dense = [
        [dic.get(k, 0) for k in keys]
        for dic in dict_lst
    ]
    return {
        "dense": dense,
        "keys": keys,
    }


def process_constraint_matrix(edges):
    # {iron: [(iron, A, B), (iron, A, C), (iron, D, E)]}
    by_ingredient = groupby(lambda x: x[0], edges)

    # {iron: {sources: [A, D], destinations: [B, C, E]}}
    transports = {
        ing: {
            "sources": list(unique(s for (i, s, d) in by_ingredient[ing])),
            "destinations": list(unique(d for (i, s, d) in by_ingredient[ing])),
        }
        for ing in by_ingredient
    }

    # [
    #   {
    #     A: A_iron_out_rate, B: B_iron_out_rate,
    #     C: -C_iron_in_rate, D: -D_iron_in_rate, E: -E_iron_in_rate,
    #   }
    # ]
    ingredient_rates = [
        {
            **{src: src.output_rate[ing] for src in data["sources"]},
            **{dest: -dest.input_rate[ing] for dest in data["destinations"]},
        }
        for (ing, data) in transports.items()
    ]

    return _densify(ingredient_rates)


def solve_milp(dense, keys, max_leak=0, max_repeat=180):
    c = np.ones(len(keys))
    A = np.array(dense)
    b_u = max_leak*np.ones(len(dense))
    b_l = np.zeros(len(dense))

    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones_like(c)
    bounds = Bounds(lb=np.ones_like(c), ub=max_repeat*np.ones_like(c))

    res = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
    )

    if res.success:
        return {
            "answer": dict(zip(keys, map(int, res.x))),
            "result": res,
        }
    else:
        raise ValueError("No solution found")


def best_milp_sequence(dense, keys):
    max_leak = 10000
    max_repeat = 500
    last = None

    try:
        soln = solve_milp(dense, keys, max_leak=max_leak, max_repeat=max_repeat)
    except ValueError:
        return
    else:
        last = soln["result"].x
        leaks = dense @ soln["result"].x
        max_leak = 0.9 * max(leaks)
        yield (max_leak, soln["answer"])

    while True:
        try:
            soln = solve_milp(
                dense,
                keys,
                max_leak=max_leak,
                max_repeat=max_repeat,
            )
        except ValueError:
            return
        else:
            if (soln["result"].x == last).all():
                return
            last = soln["result"].x
            leaks = dense @ soln["result"].x
            max_leak = 0.9 * max(leaks)
            yield (max_leak, soln["answer"])


def balance_process_tree(edges, max_leak=0, max_repeat=180):
    (dense, keys) = get(["dense", "keys"], process_constraint_matrix(edges))
    return solve_milp(dense, keys, max_leak=max_leak, max_repeat=max_repeat)


def balance_process_tree_seq(edges):
    (dense, keys) = get(["dense", "keys"], process_constraint_matrix(edges))
    return list(best_milp_sequence(dense, keys))


def net_process(kind, edges, balance_dict, **kwargs):
    net_transfer = kind.zero()

    dests = [dest for (_, _, dest) in edges]
    leaves = list(unique(src for (_, src, _) in edges if src not in dests))
    sources = [src for (_, src, _) in edges]
    roots = list(unique(dest for (_, _, dest) in edges if dest not in sources))

    # We don't support multi-root processes at this time
    if len(roots) != 1:
        raise ValueError(f"Multiple roots is an error: {roots}")

    # Add root output
    for root in roots:
        net_transfer += balance_dict[root]*root.output_rate

    # Subtract leaf input
    for leaf in leaves:
        net_transfer -= balance_dict[leaf]*leaf.input_rate

    # Subtract dangling inputs
    for dest in unique(dests):
        if dest is None:
            continue

        expected_inputs = dest.inputs.nonzero_components
        dangling = [
            inp for inp in expected_inputs
            if not any(
                True for (inp_, x, dest_) in edges
                if (inp_, dest_) == (inp, dest)
            )
        ]
        for inp in dangling:
            net_transfer -= balance_dict[dest]*dest.inputs.project(inp)

    # Add dangling outputs
    for src in unique(sources):
        if src is None:
            continue

        expected_outputs = src.outputs.nonzero_components
        dangling = [
            inp for inp in expected_outputs
            if not any(
                True for (inp_, src_, _) in edges
                if (inp_, src_) == (inp, src)
            )
        ]
        for inp in dangling:
            net_transfer += balance_dict[src]*src.outputs.project(inp)

    # Scale down duty cycles by propagating up from the leaves
    duty_cycles = {}

    for (on, src, dest) in edges:
        duty_cycles[src] = 1
        duty_cycles[dest] = 1

    sources = leaves

    while sources:
        for src in sources:
            parents = [
                (on, dest) for (on, src_, dest) in edges if src_ == src
            ]
            for (on, dest) in parents:
                provided_rate = (
                    src.output_rate[on]
                    * balance_dict[src]
                    * duty_cycles[src]
                )
                desired_rate = (
                    dest.input_rate[on]
                    * balance_dict[dest]
                    # No duty cycle as we are propagating up from leaves; duty
                    # cycles have not been established for the dest side yet
                )
                duty_cycle = min(1, provided_rate/desired_rate)
                duty_cycles[dest] = min(duty_cycle, duty_cycles[dest])
        sources = parents

    cost = kind.sum(v*k.cost_rate for (k, v) in balance_dict.items())
    seconds = 1 / duty_cycles[roots[0]]
    return Process.from_transfer(
        net_transfer,
        cost=cost,
        seconds=seconds,
        **kwargs,
    )
