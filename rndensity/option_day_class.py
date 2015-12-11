#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Class for processing of option data for one day.

"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from model_classes import BSmodel, MBSmodel, GB2model, LinearModel


class OptionDay(object):

    """Hold and process option data for a single day.

    Attributes
    ----------
    data

    Methods
    -------
    plot_smiles
    rmse
    ivrmse

    """

    def __init__(self, data=None):
        self.data = data

    def plot_smiles(self, models=[]):
        """Plot volatility smiles in the data.

        Parameters
        ----------
        models : list
            List of models to plot

        """
        days = np.sort(np.unique(self.data['days']))
        sns.set_style("white")
        colors = sns.cubehelix_palette(len(days), reverse=True)
        colors_bsm = sns.cubehelix_palette(len(days), reverse=True, rot=-.5)
        colors_gb2 = sns.cubehelix_palette(len(days), reverse=True, rot=-.75)

        for day, j in zip(days, range(len(days))):
            df = self.data[self.data['days'] == day]

            if 'BS' in models:
                bsm = BSmodel(.126, df)
                plt.plot(df['moneyness'], bsm.impvol(),
                         label=day, color=colors_bsm[j], linestyle='--')
            if 'MBS' in models:
                mbs = MBSmodel([0.516, -0.07, 0.065, 0.199, 0.055], df)
                plt.plot(df['moneyness'], mbs.impvol(),
                         label=day, color=colors_gb2[j], linestyle='--')
            if 'GB2' in models:
                gb2 = GB2model(self.calibrate([4.654, 1.466, 0], 'GB2')[1], df)
                plt.plot(df['moneyness'], gb2.impvol(),
                         label=day, color=colors_gb2[j], linestyle='--')

            if 'LM' in models:
                lm = LinearModel(self.calibrate(None, 'LM')[1], df)
                plt.plot(df['moneyness'], lm.impvol(),
                         label=day, color=colors_gb2[j], linestyle='--')

            plt.plot(df['moneyness'], df['imp_vol'],
                     label=day, color=colors[j])
            # plt.plot(df['moneyness'], df['premium'], label = d)
            # plt.plot(df['moneyness'], df['premium'] / df['vega'], label = d)
        plt.xlabel('Moneyness')
        plt.ylabel('Implied vol')
        plt.legend(title='Maturity')
        plt.show()

    def ivrmse(self, param, model_choice):
        """RMSE for the day.

        Option premium should be normalized by current underlying price!

        Returns
        -------
        error : float
            Model error for the day

        """
        args = [param, self.data]
        model = self.get_model(model_choice, *args)
        return model.ivrmse() * 100

    def calibrate(self, param_start, model_choice):
        """IVRMSE for the day.

        Option premium should be normalized by current underlying price!

        Returns
        -------
        error : float
            Model error for the day

        """
        args = [param_start, self.data]
        model = self.get_model(model_choice, *args)
        return model.calibrate(param_start)

    def get_model(self, model_choice, *args):

        if model_choice == 'BS':
            return BSmodel(*args)
        elif model_choice == 'MBS':
            return MBSmodel(*args)
        elif model_choice == 'GB2':
            return GB2model(*args)
        elif model_choice == 'LM':
            return LinearModel(*args)
        else:
            msg = 'Unknown model choice! Choose from [BS, MBS, GB2, LM].'
            raise ValueError(msg)
