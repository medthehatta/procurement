import math
from functools import partial
from functools import reduce
import re
import textwrap
from pprint import pprint

from cytoolz import sliding_window
from cytoolz import curry

from procurement import Ingredients
from process import Process

import process as proc
from process import Predicates
import ratios


mkprocess = partial(Process.from_dict, Ingredients)


assemble_basic_tech_card = mkprocess(
    {
        "outputs": "5 basic tech card",
        "inputs": "20 copper cable + 20 wood",
        "seconds": 20,
        "process": "assemble",
    }
)
assemble_copper_cable = Process(
    Ingredients.parse("8 copper cable"),
    Ingredients.parse("4 copper plate"),
    seconds=2,
    process="assemble",
)
smelt_copper_plate = Process(
    Ingredients.parse("5 copper plate"),
    Ingredients.parse("10 copper ore"),
    seconds=16,
    process="smelt",
)
mine_copper_ore = Process(
    Ingredients.parse("copper ore"),
    seconds=4,
    process="mine",
)

sequence = [
    mine_copper_ore,
    smelt_copper_plate,
    assemble_copper_cable,
    assemble_basic_tech_card,
]


fac = proc.ProcessFactory(Ingredients, Process, default_process="assembler")


sample = """


2 foo
3 bar

10 foo | fast: seconds=1
30 bar


baz | oop: seconds=2 enhanced=false
1 ping + 2 pong

oy | seconds=3 cost=2 foo + bar
3 pong

""".splitlines()


specs = list(fac.specs_from_lines(sample))


pra = list(fac.processes_from_lines(sample))


with open("/home/med/deploy/procurement/repos/krastorio/recipes.txt") as f:
    reg = proc.ProcessRegistry(fac).register_from_lines(f)


exp = proc.InteractiveRegistryExplorer(reg)





A = mkprocess({
    "outputs": "A",
    "inputs": "3 B + 4 C",
    "seconds": 1,
})
B = mkprocess({
    "outputs": "2 B",
    "inputs": "D",
    "seconds": 1,
})
C = mkprocess({
    "outputs": "C",
    "seconds": 1,
})
D = mkprocess({
    "outputs": "2 D",
    "seconds": 1,
})


edgelist = [
    ("B", B, A),
    ("C", C, A),
    ("D", D, B),
]


def interactive_build_net_process(
    desired,
    max_repeat=100,
    max_leak=0.5,
    auto_if_solo=True,
    no_recurse_part=None,
    no_recurse_path=None,
    predicate=None,
):
    # We get the edges, plus the first element is a "special" edge leading from
    # the final process to nothing.  We need this in case there is exactly one
    # process required to create the desired output: in this case there are no
    # edges, but there are processes we want to know about!
    (output, *edges) = list(
        exp.interactive_build_edgelist(
            desired,
            auto_if_solo=auto_if_solo,
            no_recurse_part=no_recurse_part,
            no_recurse_path=no_recurse_path,
            predicate=predicate,
        )
    )
    # Single process is a special case.  This cannot be balanced, it's always
    # balanced!
    if not edges:
        return [
            {
                "edges": [output],
                "balance": {output: 1},
                "leakage": 0.0,
                "net": output,
            }
        ]
    else:
        return [
            {
                "edges": edges,
                "balance": balance,
                "leakage": leakage,
                "net": proc.net_process(Ingredients, edges, balance),
            }
            for (leakage, balance) in proc.balance_process_tree_seq(edges)
        ]


i = Ingredients.lookup
p = Predicates
ibnp = interactive_build_net_process


@curry
def _try(finder, max_leak, max_repeat):
    try:
        finder(max_leak=max_leak, max_repeat=max_repeat)
    except ValueError:
        return False
    else:
        return True
