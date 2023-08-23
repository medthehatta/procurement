import math
import re
from operator import add
from operator import mul


class FormalVector:
    _ZERO = "FormalVector.zero()"
    _registry = {}
    _norm_lookup = {}

    @classmethod
    def named(cls, name, content=None):
        if name not in cls._registry:
            if content is None:
                content = name
            cls._registry[name] = cls(
                components={name: 1},
                basis={name: content},
                name=name,
            )
            cls._norm_lookup[cls._norm_name(name)] = name
        return cls._registry[name]

    @classmethod
    def _norm_name(cls, name):
        return name.lower().replace("'", "")

    @classmethod
    def lookup(cls, name):
        if name in cls._registry:
            return cls._registry[name]
        lowname = name.lower()
        if lowname in cls._norm_lookup:
            return cls._registry[cls._norm_lookup[lowname]]
        matches = [
            cls._norm_lookup[x] for x in cls._norm_lookup if lowname in x
        ]
        if len(matches) == 0:
            raise LookupError(f"Could not find entry for {name}")
        elif len(matches) == 1:
            return cls._registry[matches[0]]
        else:
            raise LookupError(f"Ambiguous matches for {name}: {matches}")

    @classmethod
    def zero(cls):
        name = cls._ZERO
        if name not in cls._registry:
            cls._registry[name] = cls(
                components={},
                basis={},
                name=name,
            )
        return cls._registry[name]

    @classmethod
    def sum(cls, vectors):
        return sum(vectors, start=cls.zero())

    @classmethod
    def componentwise(cls, func, v, w):
        components = {}
        basis = {}
        keys = set(
            list(v.components.keys()) +
            list(w.components.keys())
        )
        for k in keys:
            components[k] = func(
                v.components.get(k, 0),
                w.components.get(k, 0)
            )
            basis[k] = v.basis.get(k) or w.basis.get(k)
        return cls(components=components, basis=basis)

    @classmethod
    def dot(cls, v, w):
        x = cls.componentwise(lambda a, b: a * b, v, w)
        return sum(x.components.values())

    @classmethod
    def parse_from_list(cls, lst, populate=None, fuzzy=False):
        return cls.sum(
            cls.parse(x, populate=populate, fuzzy=fuzzy) for x in lst
        )

    @classmethod
    def parse(cls, s, populate=None, fuzzy=False):
        populate = populate or (lambda x: None)

        if fuzzy:

            def new(x):
                try:
                    return cls.lookup(x)
                except LookupError:
                    return cls.named(x, content=populate(x))

        else:

            def new(x):
                return cls.named(x, content=populate(x))

        components = [
            re.search(r"(-?\d+(?:\.\d+)?(?:\s+[*]?\s*|\s*[*]?\s+))?(.*)", y.strip()).groups()
            for y in re.split(r"\s*[+]\s*", s)
        ]
        if len(components) == 1:
            x = components[0]
            return (
                float(x[0])*new(x[1]) if x[0] else
                new(x[1])
            )
        else:
            return cls.sum(
                (
                    float(x[0])*new(x[1]) if x[0] else
                    new(x[1])
                )
                for x in components
            )

    @classmethod
    def from_triples(cls, triples, name=None):
        components = {}
        basis = {}
        for (cname, value, basis_value) in triples:
            components[cname] = value
            basis[cname] = basis_value
        return cls(components=components, basis=basis, name=name)

    def __init__(self, components=None, basis=None, name=None):
        self.name = name
        self.components = components
        self.basis = basis
        if self.components.keys() != self.basis.keys():
            raise ValueError(
                f"Component keys and basis keys do not match! "
                f"components={components.keys()}, basis={basis.keys()}"
            )

    def is_leaf(self):
        return self.components == {self.name: 1}

    @property
    def nonzero_components(self):
        return {k: v for (k, v) in self.components.items() if v != 0}

    def pure(self):
        n = len(self.nonzero_components)
        if n == 1:
            return self.triples()[0]
        else:
            raise ValueError(f"Not a pure vector (has {n} (!=1) components).")

    @property
    def content(self):
        if self.is_leaf():
            return self.basis[self.name]
        else:
            return self

    def pairs(self):
        return [(self.components[k], self.basis[k]) for k in self.components]

    def triples(self):
        return [
            (k, self.components[k], self.basis[k]) for k in self.components
        ]

    def unit(self, k):
        return FormalVector.named(k, self.basis[k])

    def project(self, k):
        return self.components[k] * self.unit(k)

    def magnitude(self):
        return math.sqrt(self.normsquare())

    def normsquare(self):
        return self.dot(self, self)

    def normalized(self):
        return (1/self.magnitude()) * self

    def reduce(self, func, agg=add, prod=mul):
        triples = iter(self.triples())
        (first_name, first_count, first_base) = next(triples)

        total = prod(first_count, func(first_base))
        for (name, count, base) in triples:
            total = agg(total, prod(count, func(base)))

        return total

    def __getitem__(self, item):
        return self.components.get(item, 0)

    def __add__(self, other):
        components = {}
        basis = {}
        keys = set(
            list(self.components.keys()) +
            list(other.components.keys())
        )
        for k in keys:
            components[k] = (
                self.components.get(k, 0) +
                other.components.get(k, 0)
            )
            basis[k] = self.basis.get(k) or other.basis.get(k)
        cls = type(self)
        return cls(components=components, basis=basis)

    def __rmul__(self, alpha):
        cls = type(self)
        return cls(
            components={k: alpha*v for (k, v) in self.components.items()},
            basis=self.basis,
        )

    def __sub__(self, other):
        return self + (-1) * other

    def __neg__(self):
        return (-1) * self

    def __bool__(self):
        return self.nonzero_components != {}

    def __repr__(self):
        if self.name:
            return self.name
        elif self.components == {}:
            return self._ZERO
        elif all(v == 0 for v in self.components.values()):
            return self._ZERO
        else:
            return " + ".join(
                f"{k}" if v == 1 else
                f"-{k}" if v == -1 else
                f"{v} {k}"
                for (k, v) in self.components.items()
                if v != 0
            )


