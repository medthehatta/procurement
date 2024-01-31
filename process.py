import math
from itertools import chain
import re

from cytoolz import sliding_window

from ratios import best_convergent


class Process:

    @classmethod
    def from_dict(cls, kind, dic):
        return cls(
            kind.parse(dic["outputs"]),
            inputs=kind.parse(dic["inputs"]) if "inputs" in dic else None,
            cost=kind.parse(dic["cost"]) if "cost" in dic else None,
            seconds=float(dic.get("seconds", 0)),
            name=dic.get("name"),
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

    @classmethod
    def parse_process(cls, s):
        stripped_lines = (line.strip() for line in s.splitlines())
        lines = [
            line for line in stripped_lines
            if line and not line.startswith("#")
        ]

        if len(lines) == 0:
            raise ValueError(f"No substantive lines in (next line):\n{s}")
        elif len(lines) == 1:
            return cls._parse_process_header(lines[0])
        elif len(lines) == 2:
            return {
                **cls._parse_process_header(lines[0]),
                "inputs": lines[1],
            }
        else:
            raise ValueError(f"Found too many lines in (next line):\n{s}")

    @classmethod
    def _parse_process_header(cls, s):
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
            attributes[k] = attributes_raw[start:end].strip()

        return {
            "outputs": product_raw,
            **attributes,
        }

    def __init__(
        self,
        outputs,
        inputs=None,
        cost=None,
        seconds=0,
        name=None,
    ):
        self.name = name or "produce"
        self.outputs = outputs
        self.inputs = inputs or type(self.outputs).zero()
        self.transfer = self.outputs - self.inputs

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

    @classmethod
    def from_chain(cls, sequence, max_value=64, name=None):
        process = chain_with_multiplicities(sequence, max_value=max_value)
        # Ew, but so it goes
        process.name = name
        return process

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


def find_multiplicity(
    process1,
    process2,
    on=None,
    max_value=None,
    max_error=None,
):
    if process1.seconds == 0 or process2.seconds == 0:
        raise ValueError(
            "Processes are not continuous, multiplicity does not make sense."
        )

    p1_out = process1.outputs.components
    p2_in = process2.inputs.components
    compatible = [x for x in p1_out if x in p2_in]

    if not compatible:
        raise ValueError(
            f"Processes have no compatible components: "
            f"{p1_out.keys()} vs {p2_in.keys()}"
        )

    if on is not None and on not in compatible:
        raise ValueError(
            f"Provided value {on=} is not a compatible component: {compatible}"
        )

    if on is None and len(compatible) > 1:
        raise ValueError(
            f"Multiple compatible components: {compatible}.  Provide `on` to "
            f"disambiguate."
        )

    if on is None and len(compatible) == 1:
        on = compatible[0]

    # Now we have an `on` which is in the list of compatible components

    actual_ratio = process2.input_rate[on] / process1.output_rate[on]

    (a, b) = best_convergent(
        actual_ratio,
        max_value=max_value,
        max_error=max_error,
    )

    error = a/b - actual_ratio

    return {
        "actual": actual_ratio,
        "ratio": (a, b),
        "error": error,
    }


def chain_multiplicities(multiplicities):
    # Need an extra element on the end so the sliding window has the last REAL
    # element in the "left" tuple position at some point.
    #
    # Any data computed regarding the "right" tuple position in that iteration
    # will be discarded (it will carry over to the next iteration, but no other
    # iterations will be processed).
    #
    with_terminal = chain(multiplicities, [(1, 1)])

    last_left2 = None
    next_coeff = 1
    left_coeff = 1
    for ((left1, left2), (right1, _)) in sliding_window(2, with_terminal):
        # The GCD will tell us how many copies of either the left or right
        # process is necessary to align b and c (which refer to the same
        # process).  mult*c/gcd is the number of copies of the LEFT process to
        # line up with b/gcd of the RIGHT process.
        #
        last_left2 = left2
        gcd = math.gcd(next_coeff*left2, right1)
        left_coeff = int(next_coeff*right1/gcd)
        next_coeff = int(next_coeff*left2/gcd)
        yield left1 * left_coeff

    yield left_coeff * last_left2


def find_multiplicities(sequence, **kwargs):
    ratio_sequence = [
        find_multiplicity(a, b, **kwargs)
        for (a, b) in sliding_window(2, sequence)
    ]
    multiplicities = [x["ratio"] for x in ratio_sequence]
    chained = list(chain_multiplicities(multiplicities))
    return {
        "counts": chained,
        "errors": [
            x["error"]*mult for (x, mult) in zip(ratio_sequence, chained)
        ],
    }


def chain_with_multiplicities(sequence, **kwargs):
    sequence = list(sequence)
    multiplicities = find_multiplicities(sequence, **kwargs)["counts"]

    m_acc = multiplicities[0]
    p_acc = sequence[0]

    for (m, p) in zip(multiplicities[1:], sequence[1:]):
        p_acc = JoinedProcess(p_acc, p, m_acc, m)
        m_acc = 1

    return p_acc
