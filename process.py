from functools import reduce
from itertools import chain
import json
import re

from cytoolz import unique
from cytoolz import get
import numpy as np
from scipy.optimize import milp
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds


class InteractiveRegistryExplorer:

    NO_SELECTION = "__NO_SELECTION__"

    def __init__(self, registry):
        self.registry = registry

    def interactive_pick(self, options):
        enumerated = list(enumerate(options, start=1))

        indexed = {str(i): opt for (i, opt) in enumerated}

        while True:
            print("Make a selection:")

            for (i, opt) in enumerated:
                print(f"{i}) {opt}")

            choice = input("Choice (A to abort, empty for 1): ").strip()

            if not choice:
                choice = "1"

            if choice.strip().lower() == "a":
                return self.NO_SELECTION

            if choice in indexed:
                return indexed[choice]
            else:
                print("Invalid choice, try again.\n")

    def interactive_build_tree(self, desired, path=None):
        path = path or []

        v_desired = desired.triples()
        if len(v_desired) != 1:
            raise NotImplementedError(
                f"Only supports single output atm. {desired}"
            )

        (part, _, _) = v_desired[0]
        path = path + [part]
        print(f"{' > '.join(path)}")
        choice = self.interactive_pick(self.registry.find(part))
        print("")
        if choice is self.NO_SELECTION:
            return None
        else:
            children = {
                inp: self.interactive_build_tree(
                    choice.inputs.project(inp),
                    path,
                )
                for inp in choice.inputs.nonzero_components
            }
            return (choice, children)


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

    def find(self, part):
        return [self._registry[i] for i in self._output_index.get(part, [])]

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
            seconds=float(dic.get("seconds", 0)),
            name=dic.get("name"),
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
        seconds=0,
        name=None,
        metadata=None,
    ):
        self.name = name or "produce"
        self.outputs = outputs
        self.inputs = inputs or type(self.outputs).zero()
        self.transfer = self.outputs - self.inputs
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

        if self.seconds:
            self.cost_rate = (1/self.seconds) * self.cost
            self.output_rate = (1/self.seconds) * self.outputs
            self.input_rate = (1/self.seconds) * self.inputs
        else:
            self.cost_rate = self.cost
            self.output_rate = self.outputs
            self.input_rate = self.inputs

        self.transfer_rate = self.output_rate - self.input_rate

    def __repr__(self):
        if self.seconds == 0:
            return f"<{type(self).__name__}: {self.name} ({self.outputs})>"
        else:
            return (
                f"<{type(self).__name__}: {self.name} "
                f"({self.output_rate})/s>"
            )

    def is_batch(self):
        return self.seconds == 0

    def is_free(self):
        return self.cost == self.kind.zero()


class JoinedProcess(Process):

    def __init__(
        self,
        process1,
        process2,
        num_process1=1,
        num_process2=1,
        name=None,
    ):
        if process1.kind is not process2.kind:
            raise ValueError(
                f"The processes do not have compatible kinds.  "
                f"They have {process1.kind} and {process2.kind} "
                f"respectively."
            )

        self.name = name

        self.kind = process1.kind
        self.p1 = process1
        self.p2 = process2
        self.num_p1 = num_process1
        self.num_p2 = num_process2

        self.seconds = (
            self.num_p1*self.p1.seconds + self.num_p2*self.p2.seconds
        )

        self.cost = self.num_p1*self.p1.cost + self.num_p2*self.p2.cost
        self.transfer = (
            self.num_p2*self.p2.transfer + self.num_p1*self.p1.transfer
        )

        self.outputs = self.kind.from_triples(
            (n, v, b) for (n, v, b) in self.transfer.triples()
            if v > 0
        )
        self.inputs = -self.kind.from_triples(
            (n, v, b) for (n, v, b) in self.transfer.triples()
            if v < 0
        )

        self.cost_rate = (
            self.num_p1*self.p1.cost_rate + self.num_p2*self.p2.cost_rate
        )
        self.transfer_rate = (
            self.num_p1*self.p1.transfer_rate +
            self.num_p2*self.p2.transfer_rate
        )

        self.output_rate = self.kind.from_triples(
            (n, v, b) for (n, v, b) in self.transfer_rate.triples()
            if v > 0
        )
        self.input_rate = -self.kind.from_triples(
            (n, v, b) for (n, v, b) in self.transfer_rate.triples()
            if v < 0
        )

    def __repr__(self):
        if self.name is None:
            return f"{self.p1} | {self.p2}"
        else:
            return super().__repr__()

    def process_tree(self):
        if isinstance(self.p1, JoinedProcess):
            left = self.p1.process_tree()
        else:
            left = (self.num_p1, self.p1)

        if isinstance(self.p2, JoinedProcess):
            right = self.p2.process_tree()
        else:
            right = (self.num_p2, self.p2)

        return [left, right]


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
    attributes_raw = re.sub(r'(\S+):', r'name=\1', attributes_raw)
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

    return dict(zip(keys, map(int, res.x)))


def balance_process_tree(edges, max_leak=0, max_repeat=180):
    (dense, keys) = get(["dense", "keys"], process_constraint_matrix(edges))
    return solve_milp(dense, keys, max_leak=max_leak, max_repeat=max_repeat)
