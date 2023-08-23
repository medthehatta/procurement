import itertools


class Combined:

    def __init__(self, *items):
        self.items = items

    def __repr__(self):
        if self.items:
            return f"{self.__class__.__name__}{self.items}"
        else:
            return f"{self.__class__.__name__}"

    def __iter__(self):
        return iter(self.items)

    def __eq__(self, other):
        return self.items == other.items

    def __getitem__(self, key):
        return self.items[key]

    @classmethod
    def flat(cls, seq):
        s = itertools.chain.from_iterable(
            x.items if isinstance(x, cls) else [x] for x in seq
        )
        lst = list(s)
        if not lst:
            return Empty
        else:
            return cls(*lst)


class Impossible_(Combined):
    pass


Impossible = Impossible_()


class Empty_(Combined):
    pass


Empty = Empty_()


class Or(Combined):

    def reduced(self):
        return Or.flat(
            x for x in self.items
            if not (x is Impossible or x is Empty)
        )


class And(Combined):

    def reduced(self):
        if any(x is Impossible or x is Empty for x in self.items):
            return Impossible
        else:
            return self


def dnf(tree):
    if isinstance(tree, Or):
        return Or.flat(dnf(x) for x in tree.items)

    elif isinstance(tree, And):
        product = itertools.product(*[dnf(y).items for y in tree.items])
        return Or.flat(And.flat(x) for x in product)

    else:
        return Or(And(tree))

