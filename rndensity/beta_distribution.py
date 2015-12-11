#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GB2 density.

"""
from __future__ import print_function, division

import numpy as np
import scipy.special as ss
import scipy.integrate as si
import matplotlib.pylab as plt
import seaborn as sns


class GB2(object):

    def __init__(self, a=1.5, b=1.03, p=.2, q=.2, T=1):

        self.p, self.q = p, q
        # Scaling
        self.a = 100 * a / T**.5
        # Martingale restriction
        self.b = np.ones_like(b) * ss.beta(p, q) \
            / ss.beta(p+1/self.a, q-1/self.a)

    def compute_mom(self):
        self.mean = self.mom(1)
        self.var = self.mom(2) - self.mean**2
        self.std = self.var**.5
        m1 = self.mom(1)
        m2 = self.mom(2)
        m3 = self.mom(3)
        m4 = self.mom(4)
        self.skew = (m3 - 3*m1*m2 + 2*m1**3) / self.var**(3/2)
        self.kurt = (m4 - 4*m1*m3 + 6*m1**2*m2 - 3*m1**4) / self.var**2 - 3

    def density(self, x):
        a, b, p, q = self.a, self.b, self.p, self.q
        return a * x**(a*p-1) / (b**(a*p) * ss.beta(p, q) \
            * (1 + (x/b)**a)**(p+q))

    def mom(self, n):
        a, b, p, q = self.a, self.b, self.p, self.q
        return b**n * ss.beta(p+n/a, q-n/a) / ss.beta(p, q)


class GB2log(GB2):

    def __init__(self, a=1.5, b=1.03, p=.2, q=.2, c=.0, T=1):
        #super(ClassName, self).__init__()
        self.N = np.max([np.array(a).size, np.array(b).size,
                         np.array(p).size, np.array(q).size,
                         np.array(T).size, np.array(c).size])
        self.p = p / 100
        self.q = q / 100
        # Scaling
        self.a = 100 * a / T**(.5 - c)
        # Martingale restriction
        self.b = np.ones_like(b) * ss.beta(p, q) \
            / ss.beta(p+1/self.a, q-1/self.a)

    def density(self, x):
        return super(GB2log, self).density(np.exp(x)) * np.exp(x)

    def mom(self, n):
        out = []
        for i in range(self.N):
            f = lambda x: x**n * self.density(x)[i]
            #print(f(np.linspace(1,2,10)))
            out.append(si.quad(f, -3, 3)[0])
            #out.append(si.quadrature(f, -3, 3)[0])
        return np.array(out)


def test_params():
    #x = np.linspace(.8, 1.2, 1e2)
    x = np.linspace(-.2, .2, 1e2)

    num = 5
    range_a = np.linspace(1, 2, num)
    range_b = np.linspace(1., 1.1, num)
    range_p = np.linspace(.1, .4, num)
    range_q = np.linspace(.1, .4, num)
    range_T = np.linspace(30, 365, num) / 365

    args_def = {'a' : range_a.mean(), 'b' : range_b.mean(),
                'p' : range_p.mean(), 'q' : range_q.mean(),
                'T' : range_T.mean()}

    ranges = {'a' : range_a, 'b' : range_b,
              'p' : range_p, 'q' : range_q, 'T' : range_T}

    fig, axes = plt.subplots(nrows = len(ranges), figsize = (6,12))
    for name, a in zip(sorted(ranges.keys()), axes):
        args = args_def.copy()
        for pi in ranges[name]:
            args[name] = pi
            f = GB2(**args).density(x)
            a.plot(x, f, label = pi)
        a.legend(title = name)
    plt.show()


def draw_moments():

    num = 10
    range_a = np.linspace(1, 3, num)
    range_c = np.linspace(-.1, .5, num)
    range_p = np.linspace(.1, .3, num) * 100
    range_q = np.linspace(.1, .3, num) * 100
    range_T = np.linspace(30, 365, num) / 365

    args_def = {'T' : range_T.mean(),
                'a' : range_a.mean(),
                'c' : range_c.mean(),
                'p' : range_p.mean(),
                'q' : range_q.mean()}

    moms = ['Std', 'Skew', 'Kurt']
    ranges = {'T' : range_T, 'a' : range_a, 'c': range_c,
              'p' : range_p, 'q' : range_q}
    params = ['T', 'a', 'c', 'p', 'q']

    fig, axes = plt.subplots(nrows=len(moms), ncols=len(ranges), sharey='row',
                             sharex='col', figsize=(7, 6))
    for col, param in enumerate(params):
        args = args_def.copy()
        args[param] = ranges[param]
        #gb2 = GB2(**args)
        gb2 = GB2log(**args)
        gb2.compute_mom()
        data = {'Std': 100 * gb2.std * (365 / args['T'])**.5,
                'Skew': gb2.skew, 'Kurt': gb2.kurt}

        for row, mom in enumerate(moms):

            axes[row, col].plot(ranges[param], data[mom], lw=4)
            axes[row, col].axvline(args_def[param], c='red', linestyle='--')
            axes[row, col].set_xlim([ranges[param].min(), ranges[param].max()])
            axes[row, 0].set_ylabel(mom)
            if mom == 'Kurt':
                axes[row, 0].set_ylim([2.5, 4.5])

        axes[-1, col].set_xlabel(param)
        axes[-1, col].locator_params(axis='x', nbins=4)

    plt.tight_layout()
    plt.savefig('../plots/paper/gb2_moments.eps',
                bbox_inches='tight', pad_inches=0)
    plt.show()


def draw_term_structure():

    num = 3
    range_a = np.linspace(4, 5, num)
    range_c = np.linspace(-.2, .2, num)
    range_T = np.linspace(.2, 1, 20)

    args_def = {'a': 3.5, 'c': 0, 'p' : 20, 'q' : 20, 'T': range_T}

    ranges = {'a' : range_a, 'c': range_c}
    params = ['a', 'c']

    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True,
                             sharex='col', figsize=(7, 3))

    for col, param in enumerate(params):
        for pvalue in ranges[param]:

            args = args_def.copy()
            args[param] = pvalue
            #gb2 = GB2(**args)
            gb2 = GB2log(**args)
            gb2.compute_mom()
            data = 100 * gb2.std * (365 / args['T'])**.5

            axes[col].plot(range_T, data, lw=4)

        axes[col].legend(ranges[param], title=param)

        axes[col].set_xlabel('T')

    axes[0].set_ylabel('Std')
    plt.tight_layout()
    plt.savefig('../plots/paper/gb2_term_structure.eps',
                bbox_inches='tight', pad_inches=.01)
    plt.show()


if __name__ == '__main__':

    sns.set_context('paper')
    sns.set_style('white')

    #test_params()
#    test_moments()

#    draw_moments()

    draw_term_structure()
