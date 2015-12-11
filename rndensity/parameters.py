#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parameter calibration and forecasting.

"""
from __future__ import print_function, division

import pandas as pd
import numpy as np
import multiprocessing as mp

from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.vector_ar.var_model import VAR

from option_day_class import OptionDay
from plotting_functions import plot_actual_predicted, plot_param_acf


class Parameters(object):

    """Calibration of model parameters.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, data, theta_start={'BS': .2},
                 models=['BS'], tbname='params'):
        """Initialize the class.

        Parameters
        ----------
        data : DataFrame
            Market data to fit with the models
        theta_start : dict
            Starting values for each model
        models : list
            The list of model names to calibrate
        tbname : str
            Name of the parameter table to store on the disk

        """
        self.data = data
        self.theta_start = theta_start
        self.models = models
        self.tbname = tbname
        self.params = None
        self.params_in = None
        self.params_out = None

    def calibrate_one_day(self, date, subset):
        """Calibrate model parameters for one day.

        Parameters
        ----------
        date : datetime
            Date
        subset : DataFrame
            Market data

        Returns
        -------
        parameters : DataFrame
            Calibrated parameters for one day

        """
        # All models
        out = []
        for model in self.models:
            theta = self.theta_start[model]
            # Calibrate
            if type(subset) == pd.DataFrame:
                subset = subset.to_records()
            pname, theta = OptionDay(subset).calibrate(theta, model)
            # Column names
            cols = pd.Index(pname, name='theta')
            # DataFrame for one model
            df = pd.DataFrame(np.atleast_2d(theta), columns=cols,
                              index=pd.Index([date], name='date'))
            # Accumulate results
            out.append(df)

        return pd.concat(out, axis=1, keys=self.models, names=['model'])

    def calibrate_all_days(self, parallel=True, debug=True):
        """Calibrate models on all days.

        Parameters
        ----------

        Returns
        -------

        Typical output:

        model          BS    MBS                                GB2
        theta       sigma      a     m1     m2     s1     s2      a      p
        date
        1996-01-10  0.129  0.725  0.004  0.092  0.152  0.031  4.642  0.017
        1996-01-17  0.126  0.687 -0.001  0.083  0.153  0.032  4.938  0.016
        1996-01-24  0.120  0.646 -0.006  0.080  0.150  0.032  5.306  0.016
        1996-01-31  0.112  0.590 -0.011  0.072  0.148  0.031  6.034  0.015
        1996-02-07  0.121  0.578 -0.010  0.067  0.167  0.032  5.891  0.013

        """
        if parallel:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.starmap(self.calibrate_one_day,
                                       self.data.groupby('date'))
        else:
            results = []
            for date, subset in self.data.groupby('date'):
                results.append(self.calibrate_one_day(date, subset))

        self.params_in = pd.concat(results)
        if not debug:
            # Save data on the disk
            fname = '../data/' + self.tbname + '_in.h5'
            self.params_in.to_hdf(fname, self.tbname)

    def forecast(self, params_raw=None, order=(3, 0), debug=True):
        """Forecast model parameters for all models.

        Parameters
        ----------
        params_raw : DataFrame
            Calibrated parameters
        order : tuple
            Orders of ARMA(p, q) model
        tbname : str
            Name of the table to be stored on the disk
        debug : bool
            Whether to rewrite data on the disk

        Returns
        -------
        params_out : DataFrame
            Calibrated and predicted parameters concatenated.
            Twice wide as the input.

        """
        if self.params_in is None:
            fname = '../data/' + self.tbname + '_in.h5'
            self.params_in = pd.read_hdf(fname, self.tbname)
        self.params_out = self.params_in.copy()
        col_names = list(self.params_out.columns.names)

        models = self.params_out.columns.get_level_values(level=col_names[0])
        models = models.unique()

        for name in models:
            # Forecast for each model
            self.params_out[name] \
                = self.forecast_out_model(self.params_out[name], order=order)

        if not debug:
            fname = '../data/' + self.tbname + '_out.h5'
            self.params_in.to_hdf(fname, self.tbname)

    def merge_inout(self, debug=True):
        """Collect all model results

        """
        if self.params_in is None:
            fname = '../data/' + self.tbname + '_in.h5'
            self.params_in = pd.read_hdf(fname, self.tbname)
        if self.params_out is None:
            fname = '../data/' + self.tbname + '_out.h5'
            self.params_out = pd.read_hdf(fname, self.tbname)

        col_names = list(self.params_out.columns.names)
        name = 'sample'
        self.params = pd.concat([self.params_in, self.params_out],
                                keys=['IN', 'OUT'], names=[name], axis=1)
        col_names.append(name)
        self.params = self.params.reorder_levels(col_names, axis=1)
#        self.params.sort_index(inplace=True, axis=1)

        if not debug:
            # Save data on the disk
            self.params.to_hdf('../data/' + self.tbname + '.h5', self.tbname)

    @staticmethod
    def forecast_out_model(data, order=(3, 0)):
        """Forecast parameters for one model.

        Parameters
        ----------
        data : DataFrame
            Parameters for one model only

        Returns
        -------
        data : DataFrame
            Predicted parameters. The same structure as input.

        """
        window = data.shape[0] // 2
        maxlags = order[0]

        out = [data[:window]]
        nobs = data.shape[0]
        for first in range(nobs - window):
            last = window + first
            if data.shape[1] == 1:
                model = ARMA(data[first:last], order=order)
                res = model.fit(method='css', disp=False)
                forecast = res.forecast(1)[0]
            else:
                model = VAR(data[first:last])
                res = model.fit(maxlags=maxlags)
                forecast = res.forecast(np.atleast_2d(data[first:last]), 1)
            out.append(forecast)

        return np.vstack(out)

    def analyze_params(self, debug=True):
        """Analyze parameters.

        """
        if self.params is None:
            fname = '../data/' + self.tbname + '.h5'
            # Load stored parameters
            self.params = pd.read_hdf(fname, self.tbname)
        # Plot actual parameters and predected ones
        plot_actual_predicted(self.params, debug)
        # Plot ACF for all parameters
        plot_param_acf(self.params, debug)
