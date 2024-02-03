from functools import reduce
from itertools import chain
import json
from uuid import uuid1
import re

from cytoolz import unique
from cytoolz import get
from cytoolz import curry
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
            if parent:
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

    def __init__(self, kind, process_kind):
        self.kind = kind
        self.process_kind = process_kind

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
            self.process_kind.from_dict(self.kind, entry)
            for entry in self.specs_from_lines(lines)
        )


class Process:

    @classmethod
    def from_dict(cls, kind, dic):
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
            process=dic.get("process"),
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
    return _densify(
        [
            {src: src.output_rate[on], dest: -dest.input_rate[on]}
            for (on, src, dest) in edges
        ]
    )


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
        return dict(zip(keys, map(int, res.x)))
    else:
        raise ValueError("No solution found")


def balance_process_tree(edges, max_leak=0, max_repeat=180):
    (dense, keys) = get(["dense", "keys"], process_constraint_matrix(edges))
    return solve_milp(dense, keys, max_leak=max_leak, max_repeat=max_repeat)


def net_transfer(kind, balance_dict):
    return kind.sum(v*k.transfer_rate for (k, v) in balance_dict.items())


def net_process(kind, edges, balance_dict, **kwargs):
    # FIXME: This is wrong, because two different processes consuming the same
    # inputs will not necessarily share those inputs evenly.  Ok for now as a
    # first pass.
    transfer = kind.sum(v*k.transfer_rate for (k, v) in balance_dict.items())
    cost = kind.sum(v*k.cost_rate for (k, v) in balance_dict.items())
    seconds = 1
    return Process.from_transfer(
        transfer,
        cost=cost,
        seconds=seconds,
        **kwargs,
    )
