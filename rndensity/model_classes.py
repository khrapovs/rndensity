#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Risk-neutral density models.

"""
from __future__ import print_function, division

import pandas as pd
import numpy as np
import scipy.stats as scs
import scipy.special as scp
import scipy.optimize as sco
import statsmodels.api as sm
import matplotlib.pylab as plt
import seaborn as sns

from impvol import blackscholes_norm, impvol_bisection

__all__ = ['BSmodel', 'MBSmodel', 'GB2model']


class Model(object):

    """Generic model class.

    """

    def __init__(self, param, data):
        """Initialize the generic model.

        """
        self.param = param
        self.data = data

    def density(self, arg):
        pass

    def plot_density(self, arg):
        plt.plot(arg, self.density(arg))
        plt.show()

    def penalty(self):
        """Penalty to impose martingale restriction.

        Returns
        -------
        penalty : float
            Penalty

        """
        return 0

    def model_error(self):
        """Model error on the case by case basis.

        Returns
        -------
        error : array
            Model error on the case by case basis

        """
        return self.premium() / self.data['premium'] - 1

    def rmse(self):
        """RMSE for the day.

        Returns
        -------
        rmse : float
            Model error for the day

        """
        return (self.model_error()**2).mean()**.5

    def impvol(self):
        """Implied Volatility.

        """
        return impvol_bisection(self.data['moneyness'], self.data['maturity'],
                                self.premium(), self.data['call'])

    def model_error_iv(self):
        """Implied Volatility error.

        """
        return self.impvol() - self.data['imp_vol']

    def ivrmse(self):
        """Implied Volatility Root Mean Squared Error.

        Returns
        -------
        ivrmse : float
            IVRMSE for the day

        """
        return (self.model_error_iv()**2).mean()**.5

    def objective(self, param):
        """Objective function for parameter calibration.

        Returns
        -------
        value : float
            Objective function value

        """
        self.__init__(param, self.data)
        # return self.rmse() + self.penalty()
        return self.rmse() + self.penalty()

    def calibrate(self, param_start):
        """Calibrate model parameters to fit the data.

        """
        method = 'L-BFGS-B'
#        method = 'Nelder-Mead'
        basin = False
        options = {'maxiter': 500, 'disp': False}
        minimizer_kwargs = {'method': method, 'bounds': self.bounds()}
        if basin:
            res = sco.basinhopping(self.objective, param_start, niter=100,
                                   disp=options['disp'],
                                   minimizer_kwargs=minimizer_kwargs)
        else:
            res = sco.minimize(self.objective, param_start, method=method,
                               options=options, bounds=self.bounds())
        return self.get_pnames(), res.x


class BSmodel(Model):

    """Black-Scholes model.

    """

    def __init__(self, param, data):
        """Initialize the model.

        """
        super().__init__(param, data)
        self.sigma = self.param

    def bounds(self):
        """Parameter bounds.

        """
        lb, ub = [None], [None]
        return list(zip(lb, ub))

    def get_name(self):
        """Get model name.
        """
        return 'BS'

    def get_pnames(self):
        """Get parameter names.

        """
        return ['sigma']

    def density(self, arg):
        """Density of the excess log-return.

        """
        mean = - self.sigma**2 * self.data['maturity']
        std = self.sigma * self.data['maturity']**.5
        return scs.norm(mean, std).pdf(arg)

    def premium(self):
        """Black-Scholes option price formula.

        Parameters
        ----------
        moneyness : float array
            Log-forward moneyness
        call : bool array
            Call/put flag. True for call, False for put

        Returns
        -------
        float array
            Option premium normalized by current underlying price

        """
        return blackscholes_norm(self.data['moneyness'], self.data['maturity'],
                                 self.sigma, self.data['call'])


class MBSmodel(Model):

    """Mixture of log-normals model.

    """

    def __init__(self, param, data):
        """Initialize the model.

        """
        super().__init__(param, data)
        mixes = (len(self.param)+1) // 3
        if np.array(self.param).ndim == 2:
            self.weights = np.vstack([self.param[:mixes-1],
                                      1-np.sum(self.param[:mixes-1], 0)])
        else:
            self.weights = np.append(self.param[:mixes-1],
                                     1-np.sum(self.param[:mixes-1]))
        self.means = self.param[mixes-1:2*mixes-1]
        self.stds = self.param[2*mixes-1:]

    def bounds(self):
        """Parameter bounds.

        """
        lb = [.4, -.2, .01, 1e-2, 1e-2]
        ub = [.8, .01, 1, 1, .14]
        return list(zip(lb, ub))

    def get_name(self):
        """Get model name.
        """
        return 'MBS'

    def get_pnames(self):
        """Get parameter names.

        """
        names, means, stds = ['a'], [], []
        mixes = (len(self.param)+1) // 3
        for i in range(mixes):
            means.append('m' + str(i+1))
            stds.append('s' + str(i+1))
        names.extend(means)
        names.extend(stds)
        return names

    def density(self, arg):
        """Density of the excess log-return.

        """
        out = 0
        for weight, mean, std in zip(self.weights, self.means, self.stds):
            scale = std * self.data['maturity']**.5
            loc = ((mean - self.data['riskfree']) *
                   self.data['maturity'] - scale**2)
            out += weight * scs.norm(loc, scale).pdf(arg)
        return out

    def premium(self):
        """Mixture of log-normals option price formula.

        Parameters
        ----------
        moneyness : float array
            Log-forward moneyness
        call : bool array
            Call/put flag. True for call, False for put

        Returns
        -------
        float array
            Option premium normalized by current underlying price

        """
        premium = 0
        for weight, mean, std in zip(self.weights, self.means, self.stds):
            shift = (self.data['riskfree'] - mean) * self.data['maturity']
            moneyness = np.array(self.data['moneyness']) + shift
            premium += weight * blackscholes_norm(moneyness,
                                                  self.data['maturity'],
                                                  std, self.data['call'])
        return premium

    def penalty(self):
        """Penalty to impose martingale restriction.

        Returns
        -------
        penalty : float
            Penalty

        """
        assert len(self.weights) == len(self.means), "Dimensions!"
        out = np.exp(self.data['riskfree'] * self.data['maturity'])
        for weight, mean in zip(self.weights, self.means):
            out -= weight * np.exp(mean * self.data['maturity'])
        return (out**2).mean()**.5


class GB2model(Model):

    """Generalized Beta of the Second kind model.

    """

    def __init__(self, param, data):
        """Initialize the model.

        """
        super().__init__(param, data)
        [param_a, param_p, param_c] = self.param
        self.param_p = param_p / 100
        self.param_a = param_a * 100 / self.data['maturity']**(.5 - param_c)
        self.param_q = .03
        self.param_b = (scp.beta(self.param_p, self.param_q) /
                        scp.beta(self.param_p+1/self.param_a,
                                 self.param_q-1/self.param_a))

    def bounds(self):
        """Parameter bounds.

        """
        lb = [1e-3, 1e-3, -1]
        ub = [10, 3, 1]
#        lb = [None, None]
#        ub = [None, None]
        return list(zip(lb, ub))

    def get_name(self):
        """Get model name.
        """
        return 'GB2'

    def get_pnames(self):
        """Get parameter names.

        """
        return ['a', 'p', 'c']

    def gb2_density(self, arg):
        """Density of the return.

        """
        return (self.param_a * arg**(self.param_a*self.param_p-1) /
                (self.param_b**(self.param_a*self.param_p) *
                scp.beta(self.param_p, self.param_q) *
                (1 + (arg / self.param_b) ** self.param_a) **
                (self.param_p+self.param_q)))

    def density(self, arg):
        """Density of the excess log-return.

        """
        return self.gb2_density(np.exp(arg)) * np.exp(arg)

    def premium(self):
        """Mixture of log-normals option price formula.

        Parameters
        ----------
        moneyness : float array
            Log-forward moneyness
        call : bool array
            Call/put flag. True for call, False for put

        Returns
        -------
        float array
            Option premium normalized by current underlying price

        """
        premium = np.zeros_like(self.data['moneyness'])
        call = self.data['call']
        if isinstance(call, bool):
            if call:
                call = np.ones_like(premium, dtype=bool)
            else:
                call = np.zeros_like(premium, dtype=bool)
        else:
            call = np.array(call)

        put = np.logical_not(call)

        arg = (1 / (1 + (np.exp(self.data['moneyness']) /
               self.param_b) ** (-self.param_a)))

        P1 = 1 - scs.beta.cdf(arg, self.param_p + 1 / self.param_a,
                              self.param_q - 1 / self.param_a)
        P2 = 1 - scs.beta.cdf(arg, self.param_p, self.param_q)

        premium[call] = (P1 - np.exp(self.data['moneyness']) * P2)[call]
        premium[put] = (np.exp(self.data['moneyness']) * (1-P2) - (1-P1))[put]

        return premium


class LinearModel(Model):

    """Linear Regression model.

    """

    def __init__(self, param, data):
        """Initialize the model.

        """
        super().__init__(param, data)

        iname = 'imp_vol' if 'imp_vol' in data.dtype.names else 'imp_vol_data'
        formula = 'np.log(' + iname + ') ~ moneyness_norm*maturity+moneyness2'
        self.model = sm.OLS.from_formula(formula, data=pd.DataFrame(self.data))

    def get_name(self):
        """Get model name.
        """
        return 'LM'

    def get_pnames(self):
        """Get parameter names.

        """
        return ['c', 'mon', 'mat', 'mm', 'm2']

    def calibrate(self, param_start):
        """Calibrate model parameters to fit the data.

        """
        return self.get_pnames(), self.model.fit().params.values

    def impvol(self):
        """Implied Volatility.

        """
        try:
            return np.exp((self.model.exog * self.param).sum(1))
        except ValueError:
            return np.exp((self.model.exog *
                          np.atleast_2d(self.param).T).sum(1))

    def premium(self):
        """Get price given implied volatility.

        Returns
        -------
        float array
            Option premium normalized by current underlying price

        """

        return blackscholes_norm(self.data['moneyness'],
                                 self.data['maturity'],
                                 self.impvol(), self.data['call'])


def test_class():
    """Try to plot various densities."""
    riskfree = .03
    maturity = 30/365
    moneyness = np.linspace(-.04, .04, 10)
    premium = np.ones_like(moneyness) * .05
    call = True
    data = {'riskfree': riskfree, 'maturity': maturity,
            'moneyness': moneyness, 'call': call, 'premium': premium}

    sigma = .13
    bsm = BSmodel(sigma, data)

    weights = [.63]
    means = [-.01, .09]
    stds = [.16, .05]
    param = weights + means + stds
    mbs = MBSmodel(param, data)

    param_a, param_p, param_c = 4, 1.5, -.05
    gb2 = GB2model([param_a, param_p, param_c], data)
    print(gb2.get_pnames())

    plt.figure()
    for model in [bsm, mbs, gb2]:
        plt.plot(moneyness, model.density(moneyness), label=model.get_name())
    plt.legend()
    plt.show()

    plt.figure()
    for model in [bsm, mbs, gb2]:
        plt.plot(moneyness, model.premium(), label=model.get_name())
    plt.legend()
    plt.show()

    plt.figure()
    for model in [bsm, mbs, gb2]:
        plt.plot(moneyness, model.impvol(), label=model.get_name())
    plt.legend()
    plt.show()

    print('BS objective function = %.4f' % bsm.objective(sigma))
    print('GB2 objective function = %.4f'
          % gb2.objective([param_a, param_p, param_c]))


def test_vector_class():
    """Try to plot various densities."""
    points = 10
    riskfree = .03
    maturity = 30/365
    moneyness = np.linspace(-.04, .04, points)
    premium = np.ones_like(moneyness) * .05
    call = True
    data = {'riskfree': riskfree, 'maturity': maturity,
            'moneyness': moneyness, 'call': call, 'premium': premium}

    sigma = np.ones(points) * .13
    bsm = BSmodel(sigma, data)

    print(bsm.premium())

    weights = np.ones(points) * .63
    means = np.vstack([np.ones(points) * -.01, np.ones(points) * .09])
    stds = np.vstack([np.ones(points) * .16, np.ones(points) * .05])
    param = np.vstack([weights, means, stds])
    mbs = MBSmodel(param, data)

    print(mbs.premium())

    param_a, param_p = np.ones(points) * 4.5, np.ones(points) * 2
    param_c = -.05 * np.ones(points)
    gb2 = GB2model([param_a, param_p, param_c], data)

    print(gb2.premium())


if __name__ == '__main__':

    sns.set_context('paper')
    sns.set_style('white')

    test_class()

    test_vector_class()
