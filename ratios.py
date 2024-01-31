import math

from cytoolz import unique


def continue_fraction(x):
    if x == math.floor(x):
        return 0
    else:
        return 1 / (x - math.floor(x))


def continuants(x, precision):
    if x >= 1:
        return (math.floor(x), list(continuants_(x - math.floor(x), precision)))
    else:
        return (0, list(continuants_(x, precision)))


def continuants_(x, precision):
    precision_loss_thresh = 1e9
    y = x
    for _ in range(precision):
        y = continue_fraction(y)
        if y > precision_loss_thresh:
            return 0
        else:
            yield math.floor(y)


def next_convergent(a2, c0, c1):
    (h0, k0) = c0
    (h1, k1) = c1
    h2 = a2 * h1 + h0
    k2 = a2 * k1 + k0
    return (h2, k2)


def convergents(conts):
    a0 = conts[0]
    as_ = conts[1]
    a1 = as_[0]
    c0 = (a0, 1)
    yield c0
    c1 = (a1*a0 + 1, a1)
    yield c1
    for a in as_[1:]:
        c2 = next_convergent(a, c0, c1)
        yield c2
        c0 = c1
        c1 = c2


def rounded_pct_error(x, y):
    return round(100 * abs(x - y) / x, 1)


def approximate_ratio(x, num=5):
    approximations = list(convergents(continuants(x, num)))
    return [(rounded_pct_error(x, a/b), a, b) for (a, b) in approximations]


def best_convergent(
    ratio,
    max_value=None,
    max_error=None,
    max_error_pct=None,
):
    provided = [
        x for x in [max_value, max_error, max_error_pct]
        if x
    ]

    if len(provided) == 0:
        raise TypeError(
            "Must provide max_value, max_error, or max_error_pct"
        )

    last = None

    for (a, b) in unique(convergents(continuants(ratio, precision=20))):
        if a == 0:
            continue

        if last is None:
            last = (a, b)

        if max_value and a > max_value:
            break

        if max_value and b > max_value:
            break

        if max_error and abs(abs(a/b) - abs(ratio)) < max_error:
            break

        if (
            max_error_pct and
            100*abs(abs(a/b) - abs(ratio))/abs(ratio) < max_error_pct
        ):
            break

        last = (a, b)

    if last is None:
        raise RuntimeError("Unknown error determining convergents")

    return last
