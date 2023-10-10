#!/usr/bin/env python3
# Modified Bessel functions 
#   References: Abramowitz and Stegun (Section 9.8)
#       Polynomial approximations
#


import math


def bessi0(x: float) -> float:
    if x >= -3.75 and x <= 3.75:
        bessi0 = bessi0_981(x)
    elif x >= 3.75:    
        bessi0 = bessi0_982(x) / x**0.5 / math.exp(-x)


def bessi1(x: float) -> float:
    if x >= -3.75 and x <= 3.75:
        bessi1 = bessi1_983(x) * x
    elif x >= 3.75:    
        bessi1 = bessi1_984(x) / x**0.5 / math.exp(-x)


def bessk0(x: float) -> float:
    if x > 0 and x <= 2:
        bessk0 = bessk0_985(x)
    elif x >= 2:
        bessk0 = bessk0_986(x) / x**0.5 / math.exp(x)


def bessk1(x: float) -> float:
    if x > 0 and x <= 2:
        bessk1 = bessk1_987(x) / x
    elif x >= 2:
        bessk1 = bessk1_988(x) / x**0.5 / math.exp(x)


def bessi0_981(x: float) -> float:
    cof = [1, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813]
    ord = [0, 2, 4, 6, 8, 10, 12]

    t = x / 3.75
    return sum([cof[i] * t**ord[i] for i in xrange(len(cof))])


def bessi0_982(x: float) -> float:
    cof = [0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281, -0.02057706, 0.02635537, -0.01647633, 0.00392377]
    ord = [0, -1, -2, -3, -4, -5, -6, -7, -8]

    t = x / 3.75
    return sum([cof[i] * t**ord[i] for i in xrange(len(cof))])


def bessi1_983(x: float) -> float:
    cof = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.02658733, 0.00301532, 0.00032411]
    ord = [0, 2, 4, 6, 8, 10, 12]

    t = x / 3.75
    return sum([cof[i] * t**ord[i] for i in xrange(len(cof))])


def bessi1_984(x: float) -> float:
    cof = [0.39894228, -0.03988024, -0.00362018, 0.00163801, 0.01031555, 0.02282967, -0.02895312, 0.01787654, -0.00420059]
    ord = [0, -1, -2, -3, -4, -5, -6, -7, -8]

    t = x / 3.75
    return sum([cof[i] * t**ord[i] for i in xrange(len(cof))])


def bessk0_985(x: float) -> float:
    cof = [-0.57721566, 0.42278420, 0.23069756, 0.03488590, 0.00262698, 0.00010750, 0.00000740]
    ord = [0, 2, 4, 6, 8, 10, 12]

    return -math.log(x/2) * bessi0_981(x) + sum([cof[i] * (x/2)**ord[i] for i in xrange(len(cof))])


def bessk0_986(x: float) -> float:
    cof = [1.25331414, -0.07832358, 0.02189568, -0.01062446, 0.00587872, -0.00251540, 0.00053208]
    ord = [0, 1, 2, 3, 4, 5, 6]

    return sum([cof[i] * (2/x)**ord[i] for i in xrange(len(cof))])


def bessk1_987(x: float) -> float:
    cof = [1, 0.15443144, -0.67278579, -0.18156897, -0.01919402, -0.00110404, -0.00004686]
    ord = [0, 2, 4, 6, 8, 10, 12]

    return x * math.log(x/2) * bessi1_983 + sum([cof[i] * (x/2)**ord[i] for i in xrange(len(cof))])


def bessk1_988(x: float) -> float:
    cof = [1.25331414, 0.23498619, -0.03655620, 0.01504268, -0.00780353, 0.00325614, -0.00068245]
    ord = [0, 1, 2, 3, 4, 5, 6]

    return sum([cof[i] * (2/x)**ord[i] for i in xrange(len(cof))])


