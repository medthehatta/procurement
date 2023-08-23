import itertools


class Combined:

    def __init__(self, items):
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

    def __len__(self):
        return len(self.items)

    @classmethod
    def new(cls, lst):
        s = itertools.chain.from_iterable(
            x.items if isinstance(x, cls) else [x] for x in lst
        )
        return cls._new(s)

    @classmethod
    def of(cls, *items):
        return cls.new(items)

    @classmethod
    def flat(cls, seq):
        lst = list(seq)
        return cls.new(lst)

    def to_dict(self):
        return {
            f"${self.__class__.__name__}": [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in self.items
            ]
        }


class Impossible_(Combined):

    def to_dict(self):
        return None


Impossible = Impossible_([])


class Empty_(Combined):

    def to_dict(self):
        return []


Empty = Empty_([])


class Or(Combined):

    @classmethod
    def _new(cls, items):
        contents = [x for x in items if not (x is Impossible or x is Empty)]
        if contents:
            return cls(contents)
        else:
            return Empty


class And(Combined):

    @classmethod
    def _new(cls, items):
        contents = []
        for x in items:
            if x is Impossible:
                return Impossible
            elif x is Empty:
                continue
            else:
                contents.append(x)

        return cls(contents)


def dnf(tree):
    if isinstance(tree, Or):
        return Or.flat(dnf(x) for x in tree.items)

    elif isinstance(tree, And):
        product = itertools.product(*[dnf(y).items for y in tree.items])
        return Or.flat(And.flat(x) for x in product)

    else:
        return Or.of(And.of(tree))

