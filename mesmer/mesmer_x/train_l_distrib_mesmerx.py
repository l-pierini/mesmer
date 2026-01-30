# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Refactored code for the training of distributions

"""

import functools
import warnings


import numpy as np
import properscoring as ps
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import basinhopping, minimize, shgo

from mesmer.core.geospatial import geodist_exact
from mesmer.mesmer_x.train_utils_mesmerx import (
    Expression,
    listxrds_to_np,
    weighted_median,
)
from mesmer.stats import gaspari_cohn

#ADDED for RX1day application
import scipy.stats as stats
from joblib import Parallel, delayed
import logging
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.interpolate import interp1d
from scipy.stats import skew, norm, truncnorm
from sklearn.metrics import mean_squared_error
from scipy.integrate import quad
import os, sys


def ignore_warnings(func):
    # adapted from https://stackoverflow.com/a/70292317
    # TODO: don't suppress all warnings

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return func(*args, **kwargs)

    return wrapper


# TODO: enable distrib class and training func for xarray objs
def xr_train_distrib(
    predictors,
    target,
    target_name,
    expr,  # TODO: replace by instance of Expression (instead of building it here)
    expr_mix,
    expr_name,
    n_jobs = 1,
    p_time = True,
    max_p_mix = None,
    option_2ndfit=False,
    option_2ndfit_mix = True, #to fit mixture gev in points with bad m.a.t.
    r_gasparicohn_2ndfit=500,
    scores_fit=["func_optim", "NLL", "BIC"],
    boundaries_params=None,
    boundaries_params_mix=None,
    options_optim=None,
    MAT_min = 0.25,
):
    """
    train in each grid point the target a conditional distribution as described by expr
    with the provided predictors. Includes extension of single-GEV fitting approach by optionally
    supporting adaptive mixture GEV models, parallel execution, and additional
    numerical safeguards.
    The length of the coefficients contain the information on whether mixture have been applied

    Parameters
    ----------
    predictors : not sure yet what data format will be used at the end.
        Assumed to be a xarray Dataset with a coordinate 'time' and 1D variable with
        this coordinate

    target : not sure yet what data format will be used at the end.
        Assumed to be a xarray Dataset with coordinates 'time' and 'gridpoint' and one
        2D variable with both coordinates

    target_name : str
        name of the variable to train

    expr : str
        See docstring of mesmer.mesmer_x.train_utils_mesmerx.Expression

    expr_mix : str
        Conditional expression defining the mixture-GEV model.

        This expression is used when mixture fitting is activated, for example in case of 
        poor single-GEV fit quality.

        total expression = p * (expr) + (1-p) * (expr_mix) with p mixture weigth

        See docstring of mesmer.mesmer_x.train_utils_mesmerx.Expression

    expr_name : str
        Name of the expression.

    option_2ndfit : boolean, default: False
        If True, will do a first fit in each gridpoint, AND THEN, will do another fit
        that uses as first guess the results of the first fit around each gridpoint.
        It helps in reducing the risk of spurious fits. Default: False.
        This is an experimental feature, that has not been extensively tested.

    n_jobs : int, default=1
        Number of parallel jobs used for grid-point-wise distribution fitting.

        A value of 1 disables parallel execution. Values greater than 1 enable
        parallel processing via joblib.

    p_time : bool, default=True
        Whether time-dependent weighting of the mixture component p(t) is applied
        during distribution fitting.

    max_p_mix : float or None, default=None
        Maximum effective value of the mixing factor p in mixture-GEV fitting. 
        (Usally around 0.997)
        If the mixture distribution component is hardly present (p close to 1),
        during the fitting procedure p might be set to 1, making the mixture 
        disappear. Setting a maximum threshold ensures that the
        mixture component can be still identified.

        If None, the maximum value can be 1.

    option_2ndfit_mix : bool, default=True
        Whether to activate the fitting of mixtrue model for points satisfying specific conditions
        (mean above threshold metric, very large shape parameter)


    r_gasparicohn_2ndfit : float
        Distance used in calculation of the correlation in Gaspari-Cohn matrix for the
        2nd fit.

    scores_fit : list, default: 'func_optim', 'NLL', 'BIC'
        After the fit, several scores can be calculated to assess the quality of the fit

        - func_optim: function optimized, as described in
          options_optim['type_fun_optim']: negative log likelihood or full conditional
          negative log likelihood
        - NLL: Negative Log Likelihood

          - BIC: Bayesian Information Criteria
          - CRPS: Continuous Ranked Probability Score (warning, takes a long time to
            compute)

    boundaries_params : sequence or None, default=None
        Bounds applied to optimization parameters for single-GEV fitting.
        For GEV, if one wants to have finite mean one needs the shape parameter smaller than 1,
        to have finite variance the shape parameter must be smaller than 1/2.

        If None, no explicit bounds are enforced.

    boundaries_params_mix : sequence or None, default=None
        Bounds applied to optimization parameters for mixture-GEV fitting.


    MAT_min : float, default=0.25
        Difference in Mean Above Threshold (MAT) between empirical and theoretical
        below which grid points are considered potentially problematic for single-GEV fitting.

        Grid points below this threshold trigger mixture-GEV fitting
        automatically.


    """

    # PREPARATION OF DATA: temporary because working on temporary format.

    # - not implementing checks on the format of predictors & target, because their
    #   current format is temporary and will be updated in MESMER v1
    # - must make 100% sure that each point of the predictors is associated with its
    #   corresponding point of the target: same ESM, same scenario, same member, same
    #   timestep, same gridpoint
    # - compiling list of scenarios, for similar order in listxrds_to_np, ensuring
    #   consistency of series of predictors & target
    
    
    # Define log file path
    log_filename = "process_log.log"
    # Redirect stdout (print statements) to a log file
    
    list_scens_pred = [item[1] for item in predictors]
    list_scens_targ = [item[1] for item in target]
    list_scens = [scen for scen in list_scens_pred if scen in list_scens_targ]
    list_scens.sort()
    gridpoints = target[0][0].gridpoint.values

    # PREPARATION OF FIT

    # getting list of inputs from predictors
    expression_fit = Expression(expr, expr_name)
    #include mixture expression
    expression_fit_mix = Expression(expr_mix, expr_name)

    # shaping predictors in current temporary format for use in np_train_distrib
    predictors_np = {
        inp: listxrds_to_np(listds=predictors, name_var=inp, forcescen=list_scens)
        for inp in list(set(expression_fit.inputs_list) | set(expression_fit_mix.inputs_list))
    }

    # preparing the Datasets of the fit:
    # here, had the choice between two options: one variable for all coefficients
    # (gridpoint, coefficient) or one variable for each coefficient (gridpoint).
    # prefers 2nd option, because more similar to MESMER, and also because will
    # facilitate future developments of MESMER-X on expressions depending on the
    # gridpoint.
    

    #add mixture coefficients
    #double the coefficients and add the mixture coefficient p to account for 
    #the possibility of fitting a mixture of 2 distribution (same type of distribs)
    modified_list = [item + 'm' for item in expression_fit_mix.coefficients_list]

    if p_time == True: #considering time dependence of mixing factor
        coefficients_mixture_list = expression_fit.coefficients_list+modified_list + ['p0'] + ['p1']

    else: #static mixing factor
        coefficients_mixture_list = expression_fit.coefficients_list+modified_list + ['p0']
    
    
    coefficients_xr = xr.Dataset()

    for coef in expression_fit.coefficients_list: #by default no mixture
        coefficients_xr[coef] = xr.DataArray(
            np.nan, coords={"gridpoint": gridpoints}, dims=("gridpoint",)
        )

    quality_xr = xr.Dataset()
    for score in scores_fit:
        if score == 'scaling_mean':
                quality_xr[score] = xr.DataArray(
                        np.full(len(gridpoints), "default_value", dtype=object),
                        coords={"gridpoint": gridpoints}, dims=("gridpoint",)
                )
        else:
            quality_xr[score] = xr.DataArray(
                np.nan, coords={"gridpoint": gridpoints}, dims=("gridpoint",)
            )

    # FIT

    print(f"Fitting the variable {target_name} with the expression {expr}:")

        
    def process_gridpoint(gp, mixture = 2, options_optim = options_optim):

        pid = os.getpid()  # Get the process ID of the current worker

        # Print which process is handling which gridpoint
        with open(log_filename, "a") as f:
            f.write(f"Gridpoint {gp} being processed by Process {pid}\n")
        """Process a single gridpoint and return results."""
        target_np = listxrds_to_np(
            listds=target,
            name_var=target_name,
            forcescen=list_scens,
            coords={"gridpoint": gp},
        )

        coefficients_np, quality_np = np_train_distrib(
            targ_np=target_np,
            pred_np=predictors_np,
            expr_fit=expression_fit,
            expr_fit_mix=expression_fit_mix,
            p_time = p_time,
            max_p_mix = max_p_mix,
            scores_fit=scores_fit,
            boundaries_params=boundaries_params,
            boundaries_params_mix=boundaries_params_mix,
            options_optim=options_optim,
            mixture=mixture,
        )


        return gp, coefficients_np, quality_np
        

    # Run gridpoints in parallel
    def process_gridpoint_wrapper(gp):
            return process_gridpoint(gp, mixture=2, options_optim = options_optim)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_gridpoint_wrapper)(gp) for gp in gridpoints
    )

    
    # Assign results back to xarray
    #coefficients_np from np_train_distrib will have different size depending on whether it uses mixture model
    for gp, coefficients_np, quality_np in results:
        # TODO: assign to .values which is much faster (in the inner loop)
        for ic, coef in enumerate(expression_fit.coefficients_list):
             coefficients_xr[coef].loc[{"gridpoint": gp}] = coefficients_np[ic]

        for score in scores_fit:
            quality_xr[score].loc[{"gridpoint": gp}] = quality_np[score]

    # SECOND FIT if required
    if option_2ndfit:
        # preparing the Datasets of the fit
        coefficients_xr2 = coefficients_xr.copy()
        quality_xr2 = quality_xr.copy()

        # remnants of MESMERv0, because stuck with its format...
        lon_l_vec = target[0][0][target_name].lon
        lat_l_vec = target[0][0][target_name].lat

        geodist = geodist_exact(lon_l_vec, lat_l_vec)

        corr_gc = gaspari_cohn(geodist / r_gasparicohn_2ndfit)

        sel_nonan = ~np.isnan(coefficients_xr[expression_fit.coefficients_list[0]])

        print(
            f"Fitting the variable {target_name} with the expression {expr}: (2nd round)"
        )

        for igp, gp in enumerate(gridpoints):
            fraction = (igp + 1) / gridpoints.size
            print(f"{fraction:0.1%}", end="\r")

            # calculate first guess, with a weighted median based on Gaspari-Cohn
            # matrix, while avoiding NaN values. Warning, weighted mean does not work
            # well, because some gridpoints may have spurious values different by orders
            # of magnitude.

            fg = np.zeros(len(expression_fit.coefficients_list))
            for ic, coef in enumerate(expression_fit.coefficients_list):
                fg[ic] = weighted_median(
                    data=coefficients_xr[coef].values[sel_nonan],
                    weights=corr_gc[igp, sel_nonan.values],
                )

            # shaping target for this gridpoint
            target_np = listxrds_to_np(
                listds=target,
                name_var=target_name,
                forcescen=list_scens,
                coords={"gridpoint": gp},
            )

            # training
            coefficients_np, quality_np = np_train_distrib(
                targ_np=target_np,
                pred_np=predictors_np,
                expr_fit=expression_fit,
                scores_fit=scores_fit,
                first_guess=fg,
                boundaries_params=boundaries_params,
                options_optim=options_optim,
            )

            # saving results
            for ic, coef in enumerate(expression_fit.coefficients_list):
                coefficients_xr2[coef].loc[{"gridpoint": gp}] = coefficients_np[ic]

            for score in scores_fit:
                quality_xr2[score].loc[{"gridpoint": gp}] = quality_np[score]

    
    # MIXTURE FIT ON DESIGNATED GRIDPOINTS
    if option_2ndfit_mix:
        # preparing the Datasets of the fit
        coefficients_xr2 = coefficients_xr.copy()
        quality_xr2 = quality_xr.copy()
        
        modified_list = [item + 'm' for item in expression_fit_mix.coefficients_list] + ['p0'] + ['p1']
        
        #I extend here the number of coefficients to account for mixture
        for coeff_mix in modified_list:
            coefficients_xr2[coeff_mix] = (('gridpoint',), np.nan * np.ones(len(gridpoints)))
        
        for score in scores_fit:
            if score == 'scaling_mean':
                quality_xr2[score+ '_mix'] = xr.DataArray(
                        np.full(len(gridpoints), "none", dtype=object),
                        coords={"gridpoint": gridpoints}, dims=("gridpoint",)
                )
            else:
                quality_xr2[score + '_mix'] = (('gridpoint',), np.nan * np.ones(len(gridpoints)))
        
        #SELECTING POINTS
        #I choose the points where the excess above the 95th percentile is larger than MAT_min (0.25) in abs value 
        #or if the tail is suspiciously large (leading to very high quantiles)
        #The idea is to select (generous) conditions to try the mixture and see if it improves the fit, if it does not, the mixture is discarded afterwards
        mask_mix = ((np.abs(quality_xr.mean_above_threshold_95)>MAT_min) | (coefficients_xr.c7<-0.25)).values
        ngridmix = np.sum(mask_mix)
        print('Number of points where mixture is applied: ', ngridmix)
        print(gridpoints[mask_mix])
        
        count_mix = 0

        #I try the mix also in points where rectification didn't work very well, just not together
        #I exclude points where I rectified
        mask_rect = ((quality_xr['rectification']<0).values) | (((quality_xr['rectification']>0).values) & ((np.abs(quality_xr.mean_above_threshold_95)>0.3)))

        gridpoints_mix = gridpoints[mask_mix & mask_rect]

        options_optim_norect = options_optim.copy()
        options_optim_norect['phys_thres_on'] = False
        
        #I have to deactivate the rectification if I want to try the mixture instead
        def process_gridpoint_wrapper_mix(gp):
            return process_gridpoint(gp, mixture=0.8, options_optim = options_optim_norect)

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_gridpoint_wrapper_mix)(gp) for gp in gridpoints_mix
        )

        #saving some metrics to help understand quality of mixture fit
        for gp, coefficients_np, quality_np in results:
            for score in scores_fit:
                quality_xr2[score+"_mix"].loc[{"gridpoint": gp}] = quality_np[score]
                
            if ("mean_above_threshold_95" in scores_fit) and ("mean_above_threshold_995" in scores_fit) and ("count_50" in scores_fit):
                #counts of points above a percentile (should match theoretical expectations)
                C50_a = np.abs(quality_xr.sel(gridpoint = gp).count_50.values.flatten().item() - 0.5)*100
                C95_a = np.abs(quality_xr.sel(gridpoint = gp).count_95.values.flatten().item() - 0.05)*100
                C05_a = np.abs(quality_xr.sel(gridpoint = gp).count_05.values.flatten().item() - 0.95)*100
                C995_a = np.abs(quality_xr.sel(gridpoint = gp).count_995.values.flatten().item() - 0.005)*100


                C50_b = np.abs(quality_np['count_50'] - 0.5)*100
                C95_b = np.abs(quality_np['count_95'] - 0.05)*100
                C995_b = np.abs(quality_np['count_995'] - 0.005)*100
                C05_b = np.abs(quality_np['count_05'] -0.95)*100

                counts_a = np.array([C05_a, C50_a, C95_a, C995_a])
                counts_b = np.array([C05_b, C50_b, C95_b, C995_b])

                #I give more importance to the fact that specific percentiles are respected
                # so I assign customized weights for the comparison
                weights_counts = np.array([0.75, 0.5, 0.35, 0.45])  

                #Introducing one single metric to understand how much I am deviating from
                #theoretical expectations
                delta_counts = np.sum((counts_b - counts_a)*weights_counts)/np.sum(weights_counts)
                n_counts = np.sum(counts_b > counts_a + 0.25)
                #TODO: think of a weighted sum of the differences to decide 
                
                if delta_counts>0:
                    continue
                
                #I only use the mixture if it actually improves the M.A.T, otherwise I just keep the previous results for the coefficients
                #Idea: keep the simpler fit in case I don't improve tails, don't want to keep unnecessary complications

                #original values of MAT, from fit without mixture, I will use them as comparison baseline
                MAT1 = quality_xr.sel(gridpoint = gp).mean_above_threshold_95.values.flatten().item()
                MAT2 = quality_xr.sel(gridpoint = gp).mean_above_threshold_995.values.flatten().item()
                
                #In some cases 99.5th percentile is completely off, so I set it to inf
                if np.isnan(MAT2):
                    MAT2 = np.inf

                func_optim0 = quality_xr.sel(gridpoint = gp).func_optim.values.flatten().item()
                #Final NLL value for first, no mixture, fit. It can be infinite if fit failed

                #Criterias to decide whether to keep mixture or not #TODO: improve, maybe decided by user how stringent
                #Currently based on empirical assessment on where mixture was worse than original fit
                #So far still necessary to do a posteriori checks in the outcomes, as some weird coefficients get past these checks 

                #If the mean above threshold metric gets worse for both 95th and 99.5th percentile or if the mixture does not converge I pass the mixture fit
                if func_optim0>0 and (np.abs(quality_np['mean_above_threshold_95']) > np.abs(MAT1)+0.075 and np.abs(quality_np['mean_above_threshold_995']) > np.abs(MAT2)+0.15) or quality_np['func_optim'] == np.inf:
                    continue

                if func_optim0>0 and np.abs(quality_np['mean_above_threshold_95']) > np.abs(MAT1)+0.3:
                    continue
                if func_optim0>0 and np.abs(quality_np['mean_above_threshold_995']) > np.abs(MAT2)+0.6:
                    continue
                if np.isnan(quality_np['mean_above_threshold_95']) or np.isnan(quality_np['mean_above_threshold_995']):
                    continue

            for ic, coef in enumerate(coefficients_mixture_list):
                coefficients_xr2[coef].loc[{"gridpoint": gp}] = coefficients_np[ic]
                    

        # Close stdout redirection
        #sys.stdout.close()

        coefficients_xr = coefficients_xr2
        quality_xr = quality_xr2
        
    # TODO: add expr as variable to coefficients_xr?
    return coefficients_xr, quality_xr


def np_train_distrib(
    targ_np,
    pred_np,
    expr_fit,
    expr_fit_mix,
    p_time = True,
    max_p_mix = 0.997, #maximum value for mixture factor initialization
    boundaries_params=None,
    boundaries_params_mix=None,
    scores_fit=["func_optim", "NLL", "BIC"],
    options_optim=None, 
    first_guess=None,
    mixture = 2, #mixture needs to be <1, =2 if no mixture selected
):

    dfit = distrib_cov(
        data_targ=targ_np,
        data_pred=pred_np,
        expr_fit=expr_fit,
        expr_fit_mix = expr_fit_mix,
        p_time = p_time,
        max_p_mix = max_p_mix,
        boundaries_params=boundaries_params,
        boundaries_params_mix=boundaries_params_mix,
        scores_fit=scores_fit,
        options_optim=options_optim,
        first_guess=first_guess,
        mixture = mixture,
    )
    

    dfit.fit()
    return dfit.coefficients_fit, dfit.quality_fit 
    #this can return double+1 (including p) the coefficients to account for mixture 
    #if the mixture is used for that gridpoint

class distrib_cov:

    def __init__(
        self,
        data_targ,
        data_pred,
        expr_fit: Expression,
        expr_fit_mix : Expression,
        data_targ_addtest=None,  # TODO: rename to data_targ_verification?
        data_preds_addtest=None,  # TODO: rename to data_preds_verification?
        threshold_min_proba=1.0e-9,
        mixture = 2,
        p_time = True,
        boundaries_params=None,
        boundaries_params_mix=None,
        boundaries_coeffs=None,
        max_p_mix = 0.997,
        first_guess=None,
        func_first_guess=None,
        scores_fit=["func_optim", "NLL", "BIC"],
        options_optim=None,  # TODO: replace by options class?
        options_solver=None,  # TODO: dito?
    ):
        """fit a conditional distribution.

        This is meant to provide flexibility in the provided expressions and robustness
        in the training. The components included in this class are:

        - (evaluation of weights for the sample)
        - tests on coefficients & parameters of the expression, in & out of sample
        - 1st optimizations for the first guess
        - 2nd optimization with minimization of the negative log likehood or full
          conditioning negative loglikelihood

        Once this class has been initialized, the go-to function is fit(). It returns
        the vector of solutions for the problem proposed.

        Parameters
        ----------
        data_targ : numpy array 1D
            Sample of the target for fit of a conditional distribution
            Normally the timeseries of the target at one gridpoint.

        data_pred : dict of 1D vectors
            Covariates for the conditional distribution. Each key must be the exact name
            of the inputs used in 'expr_fit', and the values must be aligned with the
            values in 'data_targ'.
            Normally the timeseries of the global mean predictor.

        expr_fit : class 'expression'
            Expression to train. The string provided to the class can be found in
            'expr_fit.expression'.

        expr_fit_mix : class 'expression'
            Expression to train the mixture component (same distribution type as expr_fit). 
            The string provided to the class can be found in
            'expr_fit.expression'.

        data_targ_addtest : numpy array 1D, default: None
            Additional sample of the target. The fit will not be optimized on this data,
            but will test that this data remains valid. Important to avoid points out of
            support of the distribution.

        data_preds_addtest : numpy array 1D, default: None
            Additional sample of the covariates. The fit will not be optimized on this
            data, but will test that this data remains valid. Important to avoid points
            out of support of the distribution.

        threshold_min_proba : float, default: 1e-9
            Will test that each point of the sample and added sample have their
            probability higher than this value. Important to ensure that all points are
            feasible with the fitted distribution.

        mixture : float, default:2 #TODO: probably not needed
            #Initial value for the mixture coefficient p
            #if p = 2, then no mixture fit performed, 
            #if p = 0.8, then mixture fit performed with initial guess for p=0.8


        boundaries_params : dict, default: None
            Prescribed boundaries on the parameters of the expression. Some basic
            boundaries are already provided through 'expr_fit.boundaries_params'.

        boundaries_params_mix : dict, default: None
            Prescribed boundaries on the parameters of the mixture expression. 

        boundaries_coeffs : dict, optional
            Prescribed boundaries on the coefficients of the expression. Default: None.

        max_p_mix : float or None, default=0.997
            Maximum effective value of the mixing factor p in mixture-GEV fitting.
            If the mixture distribution component is hardly present (p close to 1),
            during the fitting procedure p might be set to 1, making the mixture 
            distribution useless. Setting a maximum threshold ensures that the
            mixture component is identified.


        first_guess : numpy array, default: None
            If provided, will use these values as first guess for the first guess.

        func_first_guess : callable, default: None
            If provided, and that 'first_guess' is not provided, will be called to
            provide a first guess for the fit. This is an experimental feature, thus not
            tested.
            !! BE AWARE THAT THE ESTIMATION OF A FIRST GUESS BY YOUR MEANS COMES AT YOUR
            OWN RISKS.

        scores_fit : list of str, default: ['func_optim', 'NLL', 'BIC']
            After the fit, several scores can be calculated to assess the quality of the
            fit:

            - func_optim: function optimized, as described in
              options_optim['type_fun_optim']: negative log likelihood or full
              conditional negative log likelihood
            - NLL: Negative Log Likelihood
            - BIC: Bayesian Information Criteria
            - CRPS: Continuous Ranked Probability Score (warning, takes a long time to
              compute)

        options_optim : dict, default: None
            A dictionary with options for the function to optimize:

            * type_fun_optim: string, default: "NLL"
                If 'NLL', will optimize using the negative log likelihood. If 'fcNLL',
                will use the full conditional negative log likelihood based on the
                stopping rule. The arguments `threshold_stopping_rule`, `ind_year_thres`
                and `exclude_trigger` only apply to 'fcNLL'.

            * weighted_NLL: boolean, default: False
                If True, the optimization function will based on the weighted sum of the
                NLL, instead of the sum of the NLL. The weights are calculated as the
                inverse density in the predictors.

            * threshold_stopping_rule: float > 1, default: None
                Maximum return period, used to define the threshold of the stopping
                rule.
                threshold_stopping_rule, ind_year_thres and exclude_trigger must be used
                together.

            * ind_year_thres: np.array, default: None
                Positions in the predictors where the thresholds have to be tested.
                threshold_stopping_rule, ind_year_thres and exclude_trigger must be used
                together.

            * exclude_trigger: boolean, default: None
                Whether the threshold will be included or not in the stopping rule.
                threshold_stopping_rule, ind_year_thres and exclude_trigger must be used
                together.


        options_solver : dict, optional
            A dictionary with options for the solvers, used to determine an adequate
            first guess and for the final optimization.

            * method_fit: string, default: "Powell"
                Type of algorithm used during the optimization, using the function
                'minimize'. Prepared options: BFGS, L-BFGS-B, Nelder-Mead, Powell, TNC,
                trust-constr. The default 'Powell', is HIGHLY RECOMMENDED for its
                stability & speed.

            * xtol_req: float, default: 1e-3
                Accuracy of the fit in coefficients. Interpreted differently depending
                on 'method_fit'.

            * ftol_req: float, default: 1e-6
                Accuracy of the fit in objective.

            * maxiter: int, default: 10000
                Maximum number of iteration of the optimization.

            * maxfev: int, default: 10000
                Maximum number of evaluation of the function during the optimization.

            * error_failedfit : boolean, default: True.
                If True, will raise an issue if the fit failed.

            * fg_with_global_opti : boolean, default: False
                If True, will add a step where the first guess is improved using a
                global optimization. In theory, better, but in practice, no improvement
                in performances, but makes the fit much slower (+ ~10s).

        Notes
        -----
        This code is entirely based on the code used in MESMER-X ([1]_  and [2]_).
        However, there are some minor differences. The reasons are mostly due to
        streamlining of the code, removing deprecated features & implementing new ones.

        - Streamlining:

          - Instead of prescribing distribution & intricated inputs, using the class
            'Expression'. Shortens a lot the code and keeps the same principle.
          - The first guess has been extended to any expression. Also much shorter code.

        - Deprecated:

          - Fit of the logit of sample instead of the sample. It was used for
            expressions with sigmoids, but did not improve the fit
          - Test in the coefficients of a sigmoid to avoid simultaneous drifts. Appeared
            mostly with logits, and not feasible with class 'expression'

        - Removed
          - In tests, was checking not only whether the coefficients or parameters were
            exceeding boundaries, but also were close. Removed because of modifications
            in first guess didn't work with situations with scale=0.
          - Forcing values of certain parameters (eg mu for poisson) to be integers: to
            implement in class 'expression'?

        - Implemented:

          - Additional sample used to test the validity of the fit through tests on
            coefficients & parameters.
          - Minimum probability for the points in the sample. Useful to avoid having
            points becoming extremely unlikely, despite being in the sample.
          - Optimization may be now account as well on the stopping rule (inspired b
            https://github.com/OpheliaMiralles/pykelihood)
          - Weighting points of the sample with inverse of their density. Useful to give
            equivalent weights to the whole domain fitted.

        - Reasons for choosing that over available alternatives:
          - scipy.stats.fit: nothing about conditional distribution, all tests on
            coefficients, parameters, values AND nothing about first guess
          - pykelihood: not enough for the first guess
            (https://github.com/OpheliaMiralles/pykelihood/blob/master/pykelihood/kernels.py)
          - symfit: not enough for the first guess
            (https://symfit.readthedocs.io/en/stable/fitting_types.html#likelihood)

        .. [1] https://doi.org/10.1029%2F2022GL099012

        .. [2] https://doi.org/10.5194/esd-14-1333-2023

        """

        # preparing basic information
        self.data_targ = data_targ

        # can be different from length of predictors IF no predictors.
        self.n_sample = len(self.data_targ)

        if np.isnan(self.data_targ).any():
            raise ValueError("nan values in target")

        if np.isinf(self.data_targ).any():
            raise ValueError("infinite values in target")

        self.data_pred = data_pred
        self.pred_min = np.nanmin(data_pred['GMT_t'])


        #Since GMT is an anomaly, there could be negative values at the beginning,
        #which could affect patameter estimation
        if self.pred_min <0:
            self.data_pred['GMT_t'] = (data_pred['GMT_t'] - self.pred_min)

        if any(np.isnan(self.data_pred[pred]).any() for pred in self.data_pred):
            raise ValueError("nan values in predictors")

        if any(np.isinf(self.data_pred[pred]).any() for pred in self.data_pred):
            raise ValueError("infinite values in predictors")

        self.expr_fit = expr_fit
        self.expr_fit_mix = expr_fit_mix

        # preparing additional data
        add_test = (data_targ_addtest is not None) and (data_preds_addtest is not None)
        self.add_test = add_test

        if not self.add_test and (
            (data_targ_addtest is not None) or (data_preds_addtest is not None)
        ):
            raise ValueError(
                "Only one of `data_targ_addtest` & `data_preds_addtest` have been"
                " provided, not both of them."
            )

        self.data_targ_addtest = data_targ_addtest
        self.data_preds_addtest = data_preds_addtest

        if (threshold_min_proba <= 0) or (1 < threshold_min_proba):
            raise ValueError("`threshold_min_proba` must be in [0;1[")

        self.threshold_min_proba = threshold_min_proba
        
        self.p_time = int(p_time)



        # preparing information on boundaries
        self.boundaries_params = self.expr_fit.boundaries_parameters
        self.boundaries_params_mix = self.expr_fit_mix.boundaries_parameters
        if boundaries_params is not None:
            for param in boundaries_params:

                lower_bound = np.max(
                    [boundaries_params[param][0], self.boundaries_params[param][0]]
                )
                upper_bound = np.min(
                    [boundaries_params[param][1], self.boundaries_params[param][1]]
                )

                self.boundaries_params[param] = [lower_bound, upper_bound]
        
        #same check for mixture parameters
        if boundaries_params_mix is not None:
            for param in boundaries_params_mix:

                lower_bound = np.max(
                    [boundaries_params_mix[param][0], self.boundaries_params_mix[param][0]]
                )
                upper_bound = np.min(
                    [boundaries_params_mix[param][1], self.boundaries_params_mix[param][1]]
                )

                self.boundaries_params_mix[param] = [lower_bound, upper_bound]

        self.boundaries_coeffs = {} if boundaries_coeffs is None else boundaries_coeffs

        # preparing additional information
        self.first_guess = first_guess
        self.func_first_guess = func_first_guess

        #baseline (no mixture or rectification)
        self.n_coeffs_0 = len(self.expr_fit.coefficients_list) 

        #rectification, no mixture
        #+2 to account for the parameters of the truncated normal distribution describing 
        #the rectified portion of pdf
        self.n_coeffs_rect = len(self.expr_fit.coefficients_list) + 2  #no mixture, +2 for rectification possibility
        
        #mixture case, with possibly time evolving mixture coefficient p
        self.n_coeffs = len(self.expr_fit.coefficients_list) + len(self.expr_fit_mix.coefficients_list) + 1 + self.p_time

        if (self.first_guess is not None) and (len(self.first_guess) != self.n_coeffs) and (mixture!=2):
            raise ValueError(
                f"The provided first guess does not have the correct shape: {self.n_coeffs}"
            )

        self.scores_fit = scores_fit

        # preparing information on solver
        default_options_solver = {
            "method_fit": "Powell",
            "xtol_req": 1e-6,
            "ftol_req": 1.0e-6,
            "maxiter": 1000 * self.n_coeffs * np.log(self.n_coeffs),
            "maxfev": 1000 * self.n_coeffs * np.log(self.n_coeffs),
            "error_failedfit": False,
            "fg_with_global_opti": False,
        }

        options_solver = options_solver or {}
        if not isinstance(options_solver, dict):
            raise ValueError("`options_solver` must be a dictionary")

        # TODO: use get? (e.g. self.method_fit = options_solver.get("method_fit", "Powell")
        options_solver = default_options_solver | options_solver

        self.xtol_req = options_solver["xtol_req"]
        self.ftol_req = options_solver["ftol_req"]
        self.maxiter = options_solver["maxiter"]
        self.maxfev = options_solver["maxfev"]
        self.method_fit = options_solver["method_fit"]

        if self.method_fit not in (
            "BFGS",
            "L-BFGS-B",
            "Nelder-Mead",
            "Powell",
            "TNC",
            "trust-constr",
        ):
            raise ValueError("method for this fit not prepared, to avoid")

        xtol = {
            "BFGS": "xrtol",
            "L-BFGS-B": "gtol",
            "Nelder-Mead": "xatol",
            "Powell": "xtol",
            "TNC": "xtol",
            "trust-constr": "xtol",
        }
        ftol = {
            "BFGS": "gtol",
            "L-BFGS-B": "ftol",
            "Nelder-Mead": "fatol",
            "Powell": "ftol",
            "TNC": "ftol",
            "trust-constr": "gtol",
        }

        self.name_xtol = xtol[self.method_fit]
        self.name_ftol = ftol[self.method_fit]
        self.error_failedfit = options_solver["error_failedfit"]
        self.fg_with_global_opti = options_solver["fg_with_global_opti"]

        # preparing information on functions to optimize
        default_options_optim = dict(
            weighted_NLL=False,
            nbins_weights=20, #number of bins used for density of points estimation
            type_fun_optim="NLL",
            threshold_stopping_rule=None,
            exclude_trigger=None,
            ind_year_thres=None,
            phys_thres_on = False, #physical threshold for rectification
        )

        options_optim = options_optim or {}

        if not isinstance(options_optim, dict):
            raise ValueError("`options_optim` must be a dictionary")

        options_optim = default_options_optim | options_optim

        # preparing weights
        # TODO: move this out of init or think of more flexible bins
        self.weighted_NLL = options_optim["weighted_NLL"]
        self.bins_weights = options_optim["nbins_weights"]
        self.weights_driver = self.get_weights(n_bins_density=self.bins_weights)
        
        # Information for the rectification of the distribution around a physical (lower) boundary if needed
        # I am selecting a small threshold on the lower boundary where to separate the rectified 
        #distribution between what is considered a discrete probability mass redistributed and the continuous GEV pdf
        #NOTE: the values considered as small might be specific to precipitation values. TODO: Generalize
        if options_optim["phys_thres_on"]:
            self.phys_thres = self.rectification_threshold(min_cdf_0 = 0.005) 
            if self.phys_thres <0.75 and self.phys_thres>=0:
                self.phys_thres = np.max((self.phys_thres, 0.75))
            else:
                self.phys_thres = self.phys_thres + self.phys_thres*0.75 #a bit after the peak
            if self.phys_thres > 20: #if the threshold is high I might lose an important amount of data
                self.phys_thres = self.rectification_threshold(min_cdf_0 = 0.01)

        else:
            self.phys_thres = -np.inf

        
        #_, self.mask_cluster, self.silhouette, self.spread_clusters, self.nsmall_cluster = self._mixture_initial(silhouette_th=0.5, p_init = 0.8, dist = 1, nmin = 15)
        self.mix_init = mixture #passed as an argument 

        #This is to avoid mixture and rectification at the same time
        if (self.phys_thres > -10):
            self.mix_init = 2 #not sure that the mixture is working properly with rectification, although likelihood is ok. TODO: check
            print('Warning: rectification already in place, mixture model not applied')


        #For mixture: first estimate of presence of mixture based on clustering
        #Only first order estimate because clustering with GEV works well only if clear separation of processes
        self.mask_cluster, self.spread_clusters, self.nsmall_cluster = self._mixture_initial_nos()

        # preparing information for the stopping rule
        self.type_fun_optim = options_optim["type_fun_optim"]
        self.threshold_stopping_rule = options_optim["threshold_stopping_rule"]
        self.ind_year_thres = options_optim["ind_year_thres"]
        self.exclude_trigger = options_optim["exclude_trigger"]
        

        #First estimate of mixing factor p based either on input value or size of clusters (present/future for time dependence)
        #TODO: Generalize, now assuming GMT predictors, that I have values 3 degrees (I can have less or other predictors)
        if max_p_mix is not None:
            self.max_p_mix = max_p_mix
        else:
            p_cluster_i = np.sum(self.mask_cluster[self.data_pred['GMT_t']<=3])/np.sum(self.data_pred['GMT_t']<=3)
            p_cluster_f = np.sum(self.mask_cluster[self.data_pred['GMT_t']>3])/np.sum(self.data_pred['GMT_t']>3)
            
            if p_cluster_i > p_cluster_f and p_cluster_i >0.99: #increasing second component of the mixture
                self.max_p_mix = p_cluster_i  
                #I put a cap to p because for the first GMT point it's likely I have less mixture component, 
                # but if it starts a bit later this would create a sharp change in the quantiles
            
            else:
                self.max_p_mix = 0.999



        if self.type_fun_optim == "NLL" and (
            self.threshold_stopping_rule is not None or self.ind_year_thres is not None
        ):
            raise ValueError(
                "`threshold_stopping_rule` and `ind_year_thres` not used for"
                " `type_fun_optim='NLL'`"
                )

        if self.type_fun_optim == "fcNLL" and (
            self.threshold_stopping_rule is None or self.ind_year_thres is None
        ):
            raise ValueError(
                "`type_fun_optim='fcNLL'` needs both, `threshold_stopping_rule`"
                "  and `ind_year_thres`."
            )

    # TODO: don't do this in init. Give the user the option to either use this function
    # or give their own weigths as soon as we switch the xarray wrapper into here and
    # the user actually initialized this class themselves
    def get_weights(self, n_bins_density=20):

        if self.weighted_NLL:
            weights_driver = self._get_weights_nll(n_bins_density=n_bins_density)
        else:
            weights_driver = np.ones(self.data_targ.shape)
        # TODO: move the normalization into the function
        return weights_driver / np.sum(weights_driver)

    def _get_weights_nll(self, n_bins_density=20):
        """
        Generate weights for the sample, based on the inverse of the density of the
        predictors. More precisely, the density of the predictors is measured by a
        multidimensional histogram where each dimension is one of the predictors. The
        histogram is then smoothed by a regular grid interpolator to give the density
        of the predictors in this "predictor space". Subsequently, the weights are
        the inverse of this density of the predictors. Consequently, Samples in regions
        of this space with low densitiy will have higher weights, this is, "unusual" samples
        will have more weight.

        Parameters
        ----------
        n_bins_density : int, default: 20
            Number of bins used to calculate the density of the predictors.

        Returns
        -------
        weights_driver : numpy array 1D
            Weights for the sample, based on the inverse of the density of the
            predictors.

        Example
        -------
        TODO

        """

        # if no predictors, straightforward
        if len(self.data_pred) == 0:
            # TODO: isn't data_pred a dict and does therefore not have a shape? Yes. Also it is empty.
            # TODO: Do we want to allow no predictor?
            return np.ones(self.data_targ.shape)

        # explode data_pred dictionary into a single array for all predictors
        tmp = np.array(list(self.data_pred.values())).T

        # assessing limits on each axis
        # TODO *nan*min/max should not be necessary bc we already checked for nan values in the data?
        mn, mx = np.nanmin(tmp, axis=0), np.nanmax(tmp, axis=0)

        # TODO: at the moment bins == edges, either change bins to edges and do n_bins_density + 1
        # or change bins = n_bins_density in histogramdd
        bins = np.linspace(
            (mn - 0.05 * (mx - mn)),
            (mx + 0.05 * (mx - mn)),
            n_bins_density,
        )

        # interpolating over whole region
        gmt_hist, edges = np.histogramdd(sample=tmp, bins=bins.T)

        gmt_bins_center = [0.5 * (edge[1:] + edge[:-1]) for edge in edges]

        # TODO: add bounds_error=False, fill_value=None (extrapolates the values outside the grid)
        interp = RegularGridInterpolator(
            points=gmt_bins_center,
            values=gmt_hist,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        # evaluate interpolated density at datapoints
        density = interp(tmp)
        
        weights = 1/density #inverse of density

        return weights

    def _get_threshold(self, min_cdf_0 = 0.01):
        """
        Function to find the threshold to use for rectification in case during the fit the probability of 
        finding points below zero is not null, despite not being physically possible. 
        The threshold is found approximately by finding an empirical distribution (kde) and checking
        if there is a significant probability of having negative values if one proceeds with a standard GEV fit,
        because of the distribution laying close to the physical boundary (0 in this case).

        In case a signficant probability mass is present close to zero, as a reference separation between the rectified
        portion and the non rectified portion of the distribution (which will be treated differently) is separated
        based on the peak of the empirical distribution.

        NOTE: This is done having in mind a GEV distribution and extreme precipitation.
        TODO: Generalize
        
        min_cdf_0: tolerance for saying that there is a not null probability of finding points below 0. default: 0.01
        """

        data = self.data_targ
        kde = gaussian_kde(data)

        #I compute the CDF in 0 of the kde by summing all the gaussian kernels used to approximate the distrib
        bandwidth = np.sqrt(kde.covariance[0, 0])  # Bandwidth from the KDE

        # Compute the CDF at x = 0
        cdf_at_0 = np.mean(norm.cdf(0, loc=data, scale=bandwidth))
        
        x_grid = np.linspace(min(data), max(data), 1000)

        if cdf_at_0 < min_cdf_0:
            #if the probability of being below 0 is less than 1% then no threshold set
            return -np.inf

        else:
            # Find the maximum value and corresponding location
            kde_values = kde(x_grid)
            max_kde_value = np.max(kde_values)
            max_kde_location = x_grid[np.argmax(kde_values)]
            return max_kde_location

    def _silhouette_score(self):
        """
        Computes silhouette score to identify clustering as sign of mixture
        Used as qualitative metric but not as deciding factor.
        """
        data = np.column_stack((self.data_pred['GMT_t'], self.data_targ))

        #fit k-means to find two possible clusters within the data
        kmeans = KMeans(n_clusters =2, random_state =0) #TODO: check if improvement by changing argument values
        clusters = kmeans.fit_predict(data)
        
        #computes silhouette_score
        labels = kmeans.labels_
        silhouette = silhouette_score(data, labels)
              
        return silhouette


    def _mixture_initial_nos(self):
        """
        Function that identifies two possible clusters within the data to use as first estimate for the mixture fit.
        Returns points belonging to the bigger cluster, the relative distance between cluster centers, 
        the number of points in the smaller cluster

        """

        data = np.column_stack((self.data_pred['GMT_t'], self.data_targ))

        #as a check of the stability of the cluster estimation, I compare them with ones found by sorting the data
        
        srt_idx = np.argsort(self.data_pred['GMT_t'])
        data_pred_sorted = self.data_pred['GMT_t'][srt_idx]
        data_targ_sorted = self.data_targ[srt_idx]

        data_sorted = np.column_stack((data_pred_sorted, data_targ_sorted))

        #fit k-means to find two possible clusters within the data
        kmeans = KMeans(n_clusters =2, init = 'k-means++', random_state =42, n_init = 40) #check if improvement by changing argument values
        clusters = kmeans.fit_predict(data)
        

        #mask for identifying the two clusters:
        mask_cluster = clusters == 0

        if np.min(self.data_targ[mask_cluster]) != np.min(self.data_targ):
            #I want to take as reference the cluster containing the bulk of the distribution,
            #so the one with lower values (thinking of extreme precipitation)
            mask_cluster = clusters == 1

        #I take the distance between cluster centers and compare it with the std.deviation of the data
        spread = np.diff(kmeans.cluster_centers_[:,1])/np.std(self.data_targ)

        #I also check that I have a minimum number of points in the smaller cluster
        nsmall = np.sum(~mask_cluster)

        #fit k-means to find two possible clusters for the ordered data to see if the estimate is stable
        kmeans2 = KMeans(n_clusters =2, init = 'k-means++', random_state =42, n_init = 40) #check if improvement by changing argument values
        clusters2 = kmeans2.fit_predict(data_sorted)

        return mask_cluster, spread.item(), nsmall


    
    #TODO: remove, not needed in the approach where I decide on mixture based on mean above threshold.
    #not relying on cluster separation to identify need for mixture
    def _mixture_initial(self, silhouette_th=0.85, p_init = 0.8, dist = 4, nmin = 15):
        #functions that assign 1 to p (the mixture coefficients) if the silhouette score of two clustered data is less than 0.9
        #meaning, not enough reasons to consider the datapoints as clustered in 2
        #and assigns 0.8 to p if silouhette score is greather than silhouette_th, pointing to the need to consider the data points as belonging
        # to different clusters. In this case we fit the mixture of two distributions to account for the other cluster

        data = np.column_stack((self.data_pred['GMT_t'], self.data_targ))

        #fit k-means to find two possible clusters within the data
        kmeans = KMeans(n_clusters =2, random_state =0) #check if improvement by changing argument values
        clusters = kmeans.fit_predict(data)
        
        #mask for identifying the two clusters:
        mask_cluster = clusters == 0

        if np.min(self.data_pred['GMT_t'][mask_cluster]) != np.min(self.data_pred['GMT_t']): 
            #I want to take as reference the cluster containing the bulk of the distribution, 
            #so the one with lower values (thinking of extreme precipitation)
            mask_cluster = clusters == 1

        #computes silhouette_score
        labels = kmeans.labels_
        silhouette = silhouette_score(data, labels)

        #since the kmeans work well in the context of GEV distributions only for high separation of the clusters
        #I also check that the cluster centers are spread apart to confidently say that I have two processes
        
        #I take the distance between cluster centers and compare it with the std.deviation of the data
        spread = np.diff(kmeans.cluster_centers_[:,1])/np.std(self.data_targ)
        
        #I also check that I have a minimum number of points in the smaller cluster
        nsmall = np.sum(~mask_cluster)

        if (silhouette > silhouette_th) and (spread > dist) and (nsmall>nmin):
            p =  p_init
        else:
            p = 1
        
        return p, mask_cluster, silhouette, spread, nsmall

    def _test_coeffs_in_bounds(self, values_coeffs):

        # checking set boundaries on coefficients
        for coeff in self.boundaries_coeffs:
            bottom, top = self.boundaries_coeffs[coeff]

            # TODO: move this check to __init__
            if coeff not in self.expr_fit.coefficients_list:
                raise ValueError(
                    f"Provided wrong boundaries on coefficient, {coeff}"
                    " does not exist in expr_fit"
                )

            values = values_coeffs[self.expr_fit.list_coefficients.index(coeff)]

            if np.any(values < bottom) or np.any(top < values):
                # out of boundaries
                return False

        return True
    

    def _test_bound_params(self, distrib, a=0):
        #TODO: Not finished
        #checking that the parameters of the GEV are such that the support of the distribution lies in a certain range
        #since I am not sure imposing a support entirely positive benefits the fit in case of rectified distribution
            #for the moment I don't impose it

        # TODO: Generalize to other distributions
        # TODO: Integrate into other tests only for specific variables such as precipitation
        # TODO: Include also upper bound

        #if np.any(distrib.kwds['loc'] < a - distrib.kwds['scale']/distrib.kwds['c']):  #c is minus the shape
            #return False
            
        return True

    def _test_single_params_bounds(self, distrib):

        # checking set boundaries on parameters
        for param in self.boundaries_params:
            bottom, top = self.boundaries_params[param]
                
            # TODO: avoid using implementation detail of frozen distr of sp.stats
            param_values = distrib.kwds[param]
            
            # out of boundaries
            # TODO: why >= (and not >) or < (and not <=)?
            if np.any(param_values < bottom) or np.any(param_values >= top):
                return False
        return True

    def _test_single_params_bounds_mix(self, distrib):

        # checking set boundaries on mixture parameters
        for param in self.boundaries_params_mix:
            bottom, top = self.boundaries_params_mix[param]

            # TODO: avoid using implementation detail of frozen distr of sp.stats
            param_values = distrib.kwds[param]

            # out of boundaries
            # TODO: why >= (and not >) or < (and not <=)?
            if np.any(param_values < bottom) or np.any(param_values >= top):
                return False
        return True

    def _test_evol_params(self, distrib, data):

        # checking set boundaries on parameters
        for param in self.boundaries_params:
            bottom, top = self.boundaries_params[param]

            # TODO: avoid using implementation detail of frozen distr of sp.stats
            param_values = distrib.kwds[param]

            # out of boundaries
            # TODO: why >= (and not >) or < (and not <=)?
            if np.any(param_values < bottom) or np.any(param_values >= top):
                return False

        # test of the support of the distribution: is there any data out of the
        # corresponding support? dont try testing if there are issues on the parameters
        
        #I only check if I have points outside of the support of the distribution 
        #in the no mixture case, because with two distributions I can have points outside
        #respective ranges
        if self.mix_init == 2:
            bottom, top = distrib.support()

            # out of support
            if (
                np.any(np.isnan(bottom))
                or np.any(np.isnan(top))
                or np.any(data < bottom)
                or np.any(data > top)
            ):
                return False

        return True

    def _test_proba_value(self, distrib, data):
        # tested values must have a minimum probability of occurring, i.e. be in a
        # confidence interval
        # NOTE: DONT write 'x=data', because 'x' may be called differently for some
        # distribution (eg 'k' for poisson).

        cdf = distrib.cdf(data)
        thres = self.threshold_min_proba
        # TODO (mathause): why does this use cdf and not pdf?
        return np.all(1 - cdf >= thres) and np.all(cdf >= thres)

    def validate_coefficients(self, coefficients):
        """validate coefficients

        Validate estimated coefficients
        1. using the target data and predictors and
        2. potentially the cross-validaten data
        """

        test_coeff = self._test_coeffs_in_bounds(coefficients)

        # tests on coeffs show already that it wont work: fill in the rest with False
        if not test_coeff:
            return test_coeff, False, False, False, False

        # evaluate the distribution for the predictors and this iteration of coeffs
        distrib = self.expr_fit.evaluate(coefficients, self.data_pred)
        

        #test to check that the parameters lead to physical support
        test_bound = self._test_bound_params(distrib, a = 0)

        if not test_bound:
            return test_coeff, test_bound, False, False, False

        if self.add_test:
            distrib_add = self.expr_fit.evaluate(coefficients, self.data_preds_addtest)

        # test for the validity of the parameters
        test_param = self._test_evol_params(distrib, self.data_targ)


        if self.add_test:
            test_param &= self._test_evol_params(distrib_add, self.data_targ_addtest)

        # tests on params show already that it wont work: fill in the rest with False
        if not test_param:
            return test_coeff, test_bound, test_param, False, False

        # test for the probability of the values
        if self.threshold_min_proba is None:
            return test_coeff, test_param, True, distrib

        test_proba = self._test_proba_value(distrib, self.data_targ)

        if self.add_test:
            test_proba &= self._test_proba_value(distrib_add, self.data_targ_addtest)



        # return values for each test and the distribution that has already been
        # evaluated
        return test_coeff, test_bound, test_param, test_proba, distrib

    # suppress nan & inf warnings
    @ignore_warnings
    def find_fg(self):
        """
        compute first guess of the coefficients, to ensure convergence of the incoming
        fit.

        Motivation:
            In many situations, the fit may be complex because of complex expressions
            for the conditional distributions & because large domains in the set of
            coefficients lead to invalid fits (e.g. sample out of support).

        Criteria:
            The method must return a first guess that is ROBUST (close to the global
            minimum) & VALID (respect all conditions implemented in the tests), must be
            FLEXIBLE (any sample, any distribution & any expression).

        Method:
            1. Global fit of the coefficients of the location using derivatives, to
               improve the very first guess for the location
            2. Fit of the coefficients of the location, assuming that the center of the
               distribution should be close from its location.
            3. Fit of the coefficients of the scale, assuming that the deviation of the
               distribution should be close from its scale.
            4. Fit of remaining coefficients, assuming that the sample must be within
               the support of the distribution, with some margin.
            5. Improvement of all coefficients: better coefficients on location & scale,
               and especially estimating those on shape. Based on the Negative Log
               Likelihoog, albeit without the validity of the coefficients.
            6. Improvement of coefficients: ensuring that all points are within a likely
               support of the distribution. Two possibilities tried:
               (For 4, tried 2 approaches: based on CDF or based on NLL^n. The idea is
               to penalize very unlikely values, both works, but NLL^n works as well for
               extremely unlikely values, that lead to division by 0 with CDF)
               (step 5 still not always working, trying without?)

        Risks for the method:
            The only risk that I identify is if the user sets boundaries on coefficients
            or parameters that would reject this optimal first guess, and the domain of
            the next local minimum is far away.

        Justification for the method:
            This is a surprisingly complex problem to satisfy the criteria of
            robustness, validity & flexibility.

            a. In theory, this problem could be solved with a global optimization. Among
               the global optimizers available in scipy, they come with two types of
               requirements:
               - basinhopping: requires a first guess. Tried with first guess close from
                 optimum, good performances, but lacks in reproductibility and
                 stability: not reliable enough here. Ok if runs much longer.
               - brute, differential_evolution, shgo, dual_annealing, direct: requires
                 bounds
                 - brute, dual_annealing, direct: performances too low & too slow
                 - differential_evolution: lacks in reproductibility & stability
                 - shgo: good performances with the right sampling method, relatively
                   fast, but still adds ~10s. Highly dependent on the bounds, must not
                   be too large.
               The best global optimizer, shgo, would then require bounds that are not
               too large.

            b. The set of coefficients have only sparse valid domains. The distance
               between valid domains is often bigger than the adequate width of bounds
               for shgo.
               It implies that the bounds used for shgo must correspond to the valid
               domain that already contains the global minimum, and no other domain.
               It implies that the region of the global minimum must have already been
               identified...
               This is the tricky part, here are the different elements that I used to
               tackle this problem:
               - all combinations of local & global optimizers of scipy
               - including or not the tests for validity
               - assessment of bounds for global optimizers based on ratio of NLL or
                 domain of data_targ
               - first optimizations based on mean & spread
               - brute force approach on logspace valid for parameters (not scalable
                 with # of parameters)
             c. The only solution that was working is inspired by what the
                semi-analytical solution used in the original code of MESMER-X. Its
                principle is to use fits first for location coefficients, then scale,
                then improve.
                The steps are described in the section Method, steps 1-5. At step 5 that
                these coefficients are very close from the global minimum. shgo usually
                does not bring much at the expense of speed.
                Thus skipping global minimum unless asked.

        Warnings
        --------
        To anyone trying to improve this part:
        If you attempt to modify the calculation of the first guess, it is *absolutely
        mandatory* to test the new code on all criteria: ROBUSTNESS, VALIDITY,
        FLEXIBILITY. In particular, it is mandatory to test it for different
        situations: variables, grid points, distributions & expressions.
        """

        # preparing derivatives to estimate derivatives of data along predictors,
        # and infer a very first guess for the coefficients facilitates the
        # representation of the trends
        self.smooth_data_targ = self.smooth_data(self.data_targ)

        m, s = np.mean(self.smooth_data_targ), np.std(self.smooth_data_targ)

        ind_targ_low = np.where(self.smooth_data_targ < m - s)[0]
        ind_targ_high = np.where(self.smooth_data_targ > m + s)[0]

        pred_low = {p: np.mean(self.data_pred[p][ind_targ_low]) for p in self.data_pred}
        pred_high = {
            p: np.mean(self.data_pred[p][ind_targ_high]) for p in self.data_pred
        }

        deriv_targ = {
            p: (
                np.mean(self.smooth_data_targ[ind_targ_high])
                - np.mean(self.smooth_data_targ[ind_targ_low])
            )
            / (pred_high[p] - pred_low[p])
            for p in self.data_pred
        }

        self.fg_info_derivatives = {
            "pred_low": pred_low,
            "pred_high": pred_high,
            "deriv_targ": deriv_targ,
            "m": m,
        }

        # Initialize first guess
        if self.first_guess is None:
            self.fg_coeffs = np.zeros(self.n_coeffs_0) #only for baseline, no mixture or rectification

            # Step 1: fit coefficients of location (objective: generate an adequate
            # first guess for the coefficients of location. proven to be necessary
            # in many situations, & accelerate step 2)
            globalfit_d01 = basinhopping(
                func=self.fg_fun_deriv01, x0=self.fg_coeffs, niter=10
            )
            # warning, basinhopping tends to introduce non-reproductibility in fits,
            # reduced when using 2nd round of fits

            self.fg_coeffs = globalfit_d01.x

        else:
            # Using provided first guess, eg from 1st round of fits
            self.fg_coeffs = np.copy(self.first_guess)

        self.mem = np.copy(self.fg_coeffs)

        # Step 2: fit coefficients of location (objective: improving the subset of
        # location coefficients)
        self.fg_ind_loc = np.array(
            [
                self.expr_fit.coefficients_list.index(c)
                for c in self.expr_fit.coefficients_dict["loc"]
            ]
        )
        localfit_loc = self.minimize(
            func=self.fg_fun_loc,
            x0=self.fg_coeffs[self.fg_ind_loc],
            fact_maxfev_iter=len(self.fg_ind_loc) / self.n_coeffs_0,
            option_NelderMead="best_run",
        )
        self.fg_coeffs[self.fg_ind_loc] = localfit_loc.x

        # Step 3: fit coefficients of scale (objective: improving the subset of
        # scale coefficients)
        scale = self.expr_fit.coefficients_dict["scale"]
        self.fg_ind_sca = np.array(
            [self.expr_fit.coefficients_list.index(c) for c in scale]
        )
        if self.first_guess is None:
            # compared to all 0, better for ref level but worse for trend
            x0 = np.full(len(scale), fill_value=np.std(self.data_targ))

        else:
            x0 = self.fg_coeffs[self.fg_ind_sca]

        localfit_sca = self.minimize(
            func=self.fg_fun_sca,
            x0=x0,
            fact_maxfev_iter=len(self.fg_ind_sca) / self.n_coeffs_0,
            option_NelderMead="best_run",
        )
        self.fg_coeffs[self.fg_ind_sca] = localfit_sca.x

        # Step 4: fit other coefficients (objective: improving the subset of
        # other coefficients. May use multiple coefficients, eg beta distribution)
        other_params = [
            p for p in self.expr_fit.parameters_list if p not in ["loc", "scale"]
        ]
        if len(other_params) > 0:
            self.fg_ind_others = []
            for param in other_params:
                for c in self.expr_fit.coefficients_dict[param]:
                    self.fg_ind_others.append(self.expr_fit.coefficients_list.index(c))

            self.fg_ind_others = np.array(self.fg_ind_others)

            localfit_others = self.minimize(
                func=self.fg_fun_others,
                x0=self.fg_coeffs[self.fg_ind_others],
                fact_maxfev_iter=len(self.fg_ind_others) / self.n_coeffs_0,
                option_NelderMead="best_run",
            )
            self.fg_coeffs[self.fg_ind_others] = localfit_others.x

        # Step 5: fit coefficients using NLL (objective: improving all coefficients,
        # necessary to get good estimates for shape parameters, and avoid some local minima)
        localfit_nll = self.minimize(
            func=self.fg_fun_NLL_notests,
            x0=self.fg_coeffs,
            fact_maxfev_iter=1,
            option_NelderMead="best_run",
        )
        self.fg_coeffs = localfit_nll.x

        test_coeff, test_bound, test_param, test_proba, _ = self.validate_coefficients(
            self.fg_coeffs
        )

        if not (test_coeff and test_bound and test_param and test_proba):
            # Step 6: fit on CDF or LL^n (objective: improving all coefficients, necessary
            # to have all points within support. NB: NLL doesnt behave well enough here)
            # two potential functions:
            if False:
                # TODO: unreachable - add option or remove?
                # fit coefficients on CDFs
                fun_opti_prob = self.fg_fun_cdfs
            else:
                # fit coefficients on log-likelihood to the power n
                fun_opti_prob = self.fg_fun_LL_n

            localfit_opti = self.minimize(
                func=fun_opti_prob,
                x0=self.fg_coeffs,
                fact_maxfev_iter=1,
                option_NelderMead="best_run",
            )
            if ~np.any(np.isnan(localfit_opti.x)):
                self.fg_coeffs = localfit_opti.x

        # Step 7: if required, global fit within boundaries
        if self.fg_with_global_opti:

            # find boundaries on each coefficient
            bounds = []

            # TODO: does this assume the coeffs are ordered?
            for i_c in np.arange(self.n_coeffs_0):
                a = self.find_bound(i_c=i_c, x0=self.fg_coeffs, fact_coeff=-0.05)
                b = self.find_bound(i_c=i_c, x0=self.fg_coeffs, fact_coeff=0.05)
                vals_bounds = (a, b)

                bounds.append([np.min(vals_bounds), np.max(vals_bounds)])

            # global minimization, using the one with the best performances in this
            # situation. sobol or halton, observed lower performances with
            # implicial. n=1000, options={'maxiter':10000, 'maxev':10000})
            globalfit_all = shgo(self.func_optim, bounds, sampling_method="sobol")
            self.fg_coeffs = globalfit_all.x

    def minimize(self, func, x0, fact_maxfev_iter=1, option_NelderMead="dont_run", bounds = None):
        """
        options_NelderMead: str
            * dont_run: would minimize only the chosen solver in method_fit
            * fail_run: would minimize using Nelder-Mead only if the chosen solver in
              method_fit fails
            * best_run: will minimize using Nelder-Mead and the chosen solver in
              method_fit, then select the best results
        """
        fit = minimize(
            func,
            x0=x0,
            method=self.method_fit,
            bounds = bounds,
            options={
                "maxfev": self.maxfev * fact_maxfev_iter,
                "maxiter": self.maxfev * fact_maxfev_iter,
                self.name_xtol: self.xtol_req,
                self.name_ftol: self.ftol_req,
            },
        )

        # observed that Powell solver is much faster, but less robust. May rarely create
        # directly NaN coefficients or wrong local optimum => Nelder-Mead can be used at
        # critical steps or when Powell fails.

        if (option_NelderMead == "fail_run" and not fit.success) or (
            option_NelderMead == "best_run"
        ):
            fit_NM = minimize(
                func,
                x0=x0,
                method="Nelder-Mead",
                options={
                    "maxfev": self.maxfev * fact_maxfev_iter,
                    "maxiter": self.maxiter * fact_maxfev_iter,
                    "xatol": self.xtol_req,
                    "fatol": self.ftol_req,
                },
            )
            if (option_NelderMead == "fail_run") or (
                option_NelderMead == "best_run"
                and (fit_NM.fun < fit.fun or not fit.success)
            ):
                fit = fit_NM
        return fit

    @staticmethod
    def smooth_data(data, nn=10):
        return np.convolve(data, np.ones(nn) / nn, mode="same")

    def fg_fun_deriv01(self, x):
        params = self.expr_fit.evaluate_params(x, self.fg_info_derivatives["pred_low"])
        loc_low = params["loc"]
        params = self.expr_fit.evaluate_params(x, self.fg_info_derivatives["pred_high"])
        loc_high = params["loc"]

        deriv = {
            p: (loc_high - loc_low)
            / (
                self.fg_info_derivatives["pred_high"][p]
                - self.fg_info_derivatives["pred_low"][p]
            )
            for p in self.data_pred
        }

        return (
            np.sum(
                [
                    (deriv[p] - self.fg_info_derivatives["deriv_targ"][p]) ** 2
                    for p in self.data_pred
                ]
            )
            + (0.5 * (loc_low + loc_high) - self.fg_info_derivatives["m"]) ** 2
        )

    def fg_fun_loc(self, x_loc):
        x = np.copy(self.fg_coeffs)
        x[self.fg_ind_loc] = x_loc
        params = self.expr_fit.evaluate_params(x, self.data_pred)
        loc = params["loc"]
        return np.sum((loc - self.smooth_data_targ) ** 2)

    def fg_fun_sca(self, x_sca):
        x = np.copy(self.fg_coeffs)
        x[self.fg_ind_sca] = x_sca
        params = self.expr_fit.evaluate_params(x, self.data_pred)
        loc, sca = params["loc"], params["scale"]
        # ^ better to use that one instead of deviation, which is affected by the scale
        dev = np.abs(self.data_targ - loc)
        return np.sum((dev - sca) ** 2)

    def fg_fun_others(self, x_others, margin0=0.05):
        # preparing support
        x = np.copy(self.fg_coeffs)
        x[self.fg_ind_others] = x_others

        distrib = self.expr_fit.evaluate(x, self.data_pred)
        bot, top = distrib.support()
        val_bot = np.min(self.data_targ - bot)
        val_top = np.min(top - self.data_targ)
        # preparing margin on support
        m = np.mean(self.data_targ)
        s = np.std(self.data_targ - m)
        # optimization
        if val_bot < 0:
            # limit of val_bottom --> 0- = 1/margin0*s
            return np.exp(-val_bot) * 1 / (margin0 * s)
        elif val_top < 0:
            # limit of val_top --> 0+ = 1/margin0*s
            return np.exp(-val_top) * 1 / (margin0 * s)
        else:
            return 1 / (val_bot + margin0 * s) + 1 / (val_top + margin0 * s)

    def fg_fun_NLL_notests(self, coefficients):
        distrib = self.expr_fit.evaluate(coefficients, self.data_pred)
        self.ind_data_ok = np.arange(self.data_targ.size)
        return self.neg_loglike(coefficients)

    def fg_fun_cdfs(self, x):
        distrib = self.expr_fit.evaluate(x, self.data_pred)
        cdf = distrib.cdf(self.data_targ)

        if self.threshold_min_proba is None:
            thres = 10 * 1.0e-9
        else:
            thres = np.min([0.1, 10 * self.threshold_min_proba])

        if np.any(np.isnan(cdf)):
            return np.inf

        # DO NOT CHANGE THESE EXPRESSIONS!!
        term_low = (thres - np.min(cdf)) ** 2 / np.min(cdf) ** 2
        term_high = (thres - np.min(1 - cdf)) ** 2 / np.min(1 - cdf) ** 2
        return np.max([term_low, term_high])

    def fg_fun_LL_n(self, x, n=4):
        distrib = self.expr_fit.evaluate(x, self.data_pred)
        LL = np.sum(distrib.logpdf(self.data_targ) ** n)
        return LL

    def fg_fun_NLL_mix(self, coefficients):
        """
        NLL to find first guess of mixture component based on data in identified cluster
        """
        mask_cluster_small = ~self.mask_cluster
        data_targ_small = self.data_targ[mask_cluster_small]
        data_pred_small = self.data_pred['GMT_t'][mask_cluster_small]
        if len(self.data_pred)>1:
            pred2_lab = [key for key in self.data_pred.keys() if key != 'GMT_t'][0]
            pred2 = self.data_pred[pred2_lab]
            data_pred_small_2 = self.data_pred[pred2_lab][mask_cluster_small]
            distrib = self.expr_fit_mix.evaluate(coefficients, {'GMT_t': data_pred_small, pred2_lab: data_pred_small_2})
        else:
            distrib = self.expr_fit_mix.evaluate(coefficients, {'GMT_t': data_pred_small})
        # compute loglikelihood
        if self.expr_fit_mix.is_distrib_discrete:
            LL = distrib.logpmf(data_targ_small)
        else:
            LL = distrib.logpdf(data_targ_small)

        # weighted sum of the loglikelihood
        value = np.sum((self.weights_driver[mask_cluster_small] * LL))
        
        if np.isnan(value):
            return -np.inf
        else:
            return -value #negative loglikelihood

    def find_bound(self, i_c, x0, fact_coeff):
        # could be accelerated using dichotomy, but 100 iterations max are fast enough
        # not to require to make this part more complex.
        x, iter, itermax, test = np.copy(x0), 0, 100, True
        while test and (iter < itermax):
            test_c, test_b, test_p, test_v, _ = self.validate_coefficients(x)
            test = test_c and test_b and test_p and test_v
            x[i_c] += fact_coeff * x[i_c]
            iter += 1
        return x[i_c]
    
    

    # OPTIMIZATION FUNCTIONS & SCORES
    def func_optim(self, coefficients) :
        # check whether these coefficients respect all conditions: if so, can compute a
        # value for the optimization
        
        #MIXTURE CASE
        if len(coefficients) == self.n_coeffs: 
            coeffs = coefficients[:len(self.expr_fit.coefficients_list)] #first component of mix
            
            #mixing factor coefficient p
            p0 = coefficients[-1 - self.p_time] #if time dependence is active or not
            if self.p_time == 1:
                p1 = coefficients[-1]
            else:
                p1 = 0

            #mixture distribution coefficients
            coeffs2 = coefficients[len(self.expr_fit.coefficients_list):-1- self.p_time] #second component
            distrib2 = self.expr_fit_mix.evaluate(coeffs2, self.data_pred)
            test_distrib2 = self._test_single_params_bounds_mix(distrib2)
            
            # --- Some parameter checks for mixture, to constrain additional parameter estimation

            #Checking that the median of the second distribution in the mixture is withing the data range
            #I want this to be False, if True reject
            test_median2 = np.any(distrib2.ppf(0.5)>np.max(self.data_targ)) 

            #For very small GMT values, I have very few data points (which then weight a lot in the fit)
            #and don't immediately show the mixture component given the small sample
            #this would initialize p =1 based on the first points, 
            #but then the percentiles would change abruptly for larger GMT values given the more datapoints

            #Here checking:
            # - the mixture weight p = p0 + p1 * GMT is less than 1
            # - the mixture weight is always more than 0.1 (Being distrib1 the main component, I don't want it to disappear over time) #TODO: Not the most general case
            # - p0 is smaller than the maximum value provided as input (default 0.997)
            #I want this to be False, if True reject
            test_pt = np.any((p0+self.data_pred['GMT_t']*p1)[self.data_pred['GMT_t']>=0] >=1) or np.any(p0+self.data_pred['GMT_t']*p1 <0.1) or p0>self.max_p_mix

            #Assuming a GEV for the mixture component, if shape>=1, the mean of the distribtuion is infinite
            #if shape>=0.5, the variance is undefined #NOTE: GEV-specific
            #Here I check c<-0.5 (in practive -0.4 to avoid huge variance), with an additional caveat:
            # - if two components are very well separated, the mixture component is at high intensities 
            #   and large tails would produce overestimated events, so I constrain the shape below 0.2 #TODO: Check again if necessary, generalize
            #I want this to be False, reject if True
            if 'c' in self.expr_fit_mix.parameters_list:
                if p0>0.9: #well separated cluster (mixture at high values)
                    test_mean = np.any(distrib2.kwds['c']<=-0.2)
                else:
                    test_mean = np.any(distrib2.kwds['c']<=-0.4)
            else:
                test_mean = False
            
            #If I have time evolution on the scale: sigma = sigma_0 + GMT * sigma_1
            # I want sigma to always be positive
            # Checking that the scale does not shrink too much over time, pointing to something weird in the fit #TODO: formalize?
            #I want this to be False
            if len(self.expr_fit_mix.coefficients_dict['scale'])>1:
                test_scale_evol = np.any(distrib2.kwds['scale'] / distrib2.kwds['scale'][0] < 0.1)    
            else:
                test_scale_evol = False

            #Checking that the support of the two distributions individually is above zero
            #I just check whether the CDF of the two distributions in zero is more than 1%        
            #I want this to be False 
            #Note: Having in mind absolute values of precipitation (strictly positive values)
            #Note: I apply mixture only when I don't have rectification
            distrib1 = self.expr_fit.evaluate(coeffs, self.data_pred)
            
            bottom1 = distrib1.ppf(1e-6) #not used here
            bottom2 = distrib2.ppf(1e-6) #not used here


            if test_distrib2 == False or np.any(distrib1.cdf(0)>1e-2) or np.any(distrib2.cdf(0)>1e-6) or test_scale_evol == True or test_mean == True or test_pt == True:
                return np.inf


        #NO MIXTURE CASE      
        else: 
            coeffs = coefficients
            p0 = 1 #no mixture
            p1 = 0
            test_pt = False
            test_median2 = False


        #Checks on either main component of mixture (if present) or just the one distribution
        #I want these to be True, reject otherwise
        test_coeff, test_bound, test_param, test_proba, distrib = self.validate_coefficients(
            coeffs
        )
        
        distrib0 = self.expr_fit.evaluate(coeffs, self.data_pred) 
        bottom = distrib0.ppf(1e-6) #practical boundary, values with very small probability

        #Checking the support of the main distribution
        #NOTE: I only check the support without rectification, 
            #When I rectify, the GEV is allowed to have negative values, which I then remove and ridistribute close to zero.       
        #I want this to be True, otherwise reject
        if np.any(distrib0.cdf(0) >1e-2) and self.phys_thres<0:
            test_supp_precip = False
        else:
            test_supp_precip = True
        
        #Checking the shape parameter of the main distribution to have finite mean 
        #TODO: Assumes GEV, generalize
        #I want this to be False, reject otherwise
        test_mean_0 = np.any(distrib0.kwds['c']<=-0.9) 

        #TODO: make the logic of tests uniform (I want them all True or all False)
        if test_coeff and test_bound and test_param and test_proba and test_pt==False and test_median2 == False and test_supp_precip and test_mean_0 == False:
            # check for the stopping rule
            if self.type_fun_optim == "fcNLL":
                # will apply the stopping rule: splitting data_fit into two sets of data
                # using the given threshold
                self.ind_data_ok, self.ind_data_stopped = self.stopping_rule(distrib)
            else:
                self.ind_data_ok = slice(None)

            # compute negative loglikelihood
            #NOTE: switched to function based on coefficients, not distribution
            #the coefficients inform whether I have mixture or not
            NLL = self.neg_loglike(coefficients)

            # eventually compute full conditioning
            if self.type_fun_optim == "fcNLL":
                FC = self.fullcond_thres(distrib)
                optim = NLL + FC

            #Tried to optimize based on Anderson-Darling metric for skewed distributions
            #to get tails right but not working too well
            elif self.type_fun_optim == "ADR":
                ADR = self.adr_metric(distrib)
                #print(ADR)
                optim = NLL + ADR
            else:
                optim = NLL

        # returns value for optimization
        # TODO: merge with previous if
        if test_coeff and test_bound and test_param and test_proba and test_pt==False and test_median2 == False and test_supp_precip and test_mean_0 == False:
            return optim
        else:
            # something wrong: returns a blocking value
            return np.inf
 

    def neg_loglike(self, coefficients):
        return -self.loglike(coefficients)
    
    #Rectification Case
    def neg_loglike_rect(self, coefficients_rect):
        """
        Used to find a continuous pdf approximating the discrete probability mass in the rectification case.
        Separate from main fitting procedure.
        """
        return -self.loglike_rect(coefficients_rect)

    
    def loglike(self, coefficients):
        """
        Loglikelihood used to find the parameters of the distribution(s), 
        including mixture and rectification options.
        If mixture the coefficients include 1st distribution, 2nd distribution, mixture factor p
        If rectification, returns the coefficients of the (one) distribution above threshold a.
        """

        #Rectification threshold used to separate data (accounts for margin, not exactly 0):
        a = self.phys_thres
        # Above: gets fitted as usual
        # Below: Redistributed probability mass, 
        #   approximated as truncated normal to get one whole continuous PDF instead of discrete+continuous

        #MIXTURE CASE
        #get relevant coefficients
        if len(coefficients) == self.n_coeffs: 
            p0 = coefficients[-1- self.p_time]

            if self.p_time == 1: #time-varying mixture factor p
                p1 = coefficients[-1]
            else:
                p1 = 0

            n = len(self.expr_fit.coefficients_list)
            coeffs1 = coefficients[:n] #first component
            coeffs2 = coefficients[n:-1- self.p_time]  #second component

        #STANDARD (ONE DISTRIBUTION) CASE
        else:
            coeffs1 = coefficients

        data = self.data_targ[self.ind_data_ok]
        pred = self.data_pred['GMT_t'][self.ind_data_ok] #TODO: Generalize to non GMT predictors
        if len(self.data_pred)>1:
            pred2_lab = [key for key in self.data_pred.keys() if key != 'GMT_t'][0]
            pred2 = self.data_pred[pred2_lab][self.ind_data_ok]

        #separate data points based on rectification threshold
        is_boundary = (data < a) #close to physical boundary (redistributed points)
        is_interior = (data >= a) #above rectification threshold

        #POINTS ABOVE THRESHOLD
        #computes loglikelihood in the standard way for these points
        if len(self.data_pred)==1: 
            distrib = self.expr_fit.evaluate(coeffs1, {'GMT_t': pred[is_interior]})
        else:
            distrib = self.expr_fit.evaluate(coeffs1, {'GMT_t': pred[is_interior], pred2_lab: pred2[is_interior]})

        #mixture case
        if len(coefficients) == self.n_coeffs:
            if len(self.data_pred)==1:
                distrib2 = self.expr_fit_mix.evaluate(coeffs2, {'GMT_t': pred[is_interior]})
            else:
                distrib2 = self.expr_fit_mix.evaluate(coeffs2, {'GMT_t': pred[is_interior], pred2_lab: pred2[is_interior]})
         
        if self.expr_fit.is_distrib_discrete:
            LL_interior = distrib.logpmf(data[is_interior]) #TODO: update the scalar case
        else:
            #mixture case
            if len(coefficients) == self.n_coeffs: 
                p = p0 + p1*pred[is_interior]
                LL_interior = np.log(p*distrib.pdf(data[is_interior]) + (1-p)* distrib2.pdf(data[is_interior]))
            #standard case
            else:
                LL_interior = distrib.logpdf(data[is_interior])
                
        #weighted sum of the interior part based on density of points
        value_interior = np.sum((self.weights_driver[is_interior] * LL_interior))


        #POINTS BELOW THRESHOLD
        #The contributions of the points below threshold is computed as the log of the CDF in the threshold
        #assuming that the threshold is small, the CDF in the boundary should summarize the probability of all the points 
        #between 0 and a.
        #TODO: to try to be more precise I could approximate this part with truncated normal pdf and compute as above
        if len(self.data_pred)==1:
            distrib = self.expr_fit.evaluate(coeffs1, {'GMT_t': pred[is_boundary]})
        else:
            distrib = self.expr_fit.evaluate(coeffs1, {'GMT_t': pred[is_boundary], pred2_lab: pred2[is_boundary]})

        #mixture case
        if len(coefficients) == self.n_coeffs: 
            p = p0 + p1*pred[is_boundary]
            if len(self.data_pred)==1:
                distrib2 = self.expr_fit_mix.evaluate(coeffs2, {'GMT_t': pred[is_boundary]})
            else:
                distrib2 = self.expr_fit_mix.evaluate(coeffs2, {'GMT_t': pred[is_boundary], pred2_lab: pred2[is_boundary]})
            logcdf = np.log(p*distrib.cdf(a) + (1-p)* distrib2.cdf(a))
        #standard case
        else: 
            logcdf = distrib.logcdf(a)
        
        #TODO: double check that I can use the same density weights to normalize the boundary part with the cdf
        value_boundary =  np.sum(self.weights_driver[is_boundary] *logcdf)  

        #Total value of LogLikelihood 
        value = value_interior + value_boundary

        if np.isnan(value):
            return -np.inf
        else:
            return value


    def loglike_rect(self, coefficients_rect):
        """
        Loglikelihood used to fit of a truncated normal distribution to approximate the
        discrete probability mass between 0 and the empirical rectification threshold a.

        """
        a = self.phys_thres
        lb = 0
        coeffs_rect = coefficients_rect
        data = self.data_targ[self.ind_data_ok]
        pred = self.data_pred['GMT_t'][self.ind_data_ok]
        if len(self.data_pred)>1:
            pred2_lab = [key for key in self.data_pred.keys() if key != 'GMT_t'][0]
            pred2 = self.data_pred[pred2_lab][self.ind_data_ok]
        #define points close to the boundary and points inside the boundary
        is_boundary = (data < a)
        is_interior = (data >= a)

        LL_boundary =np.log(truncnorm.pdf(data[is_boundary], (lb - coeffs_rect[0])/coeffs_rect[1], (a - coeffs_rect[0])/coeffs_rect[1], loc = coeffs_rect[0], scale = coeffs_rect[1]))
        
        value = np.sum(LL_boundary)

        if np.isnan(value):
            return -np.inf
        else:
            return value

    def adr_metric(self, distrib):
        #computes the Anderson-Darling statistic for the right tail (ADR), to be minimized
        #Not good performance, later discarded
        z = distrib.cdf(self.data_targ) #CDF
        n = len(self.data_targ)

        #sort the CDF values in ascending order
        z_sorted = np.sort(z)

        adr = (
        n / 2
        - 2 * np.sum(z_sorted)
        - (1 / n) * np.sum((2 * np.arange(1, n + 1) - 1) * np.log(1 - z_sorted[::-1]))
        )

        return adr

    def mean_above_threshold(self, distrib, distrib2 = None, q=0.95, q_arr=None):
        """
        Computes the empirical expected value above a certain quantile (e.g. 0.95), called mean above threshold (MAT) 
        and compares it with the theoretical value of the distribution obtained from first fit (without Mixture).

        Returns MAT_empirical - MAT_theoretical

        This metric is used to identify gridpoints where there is a probability mass in the high tail of the distribution
        that is not well captured by a single-distribution fit.
        Used as criteria to try mixture fit.

        Compute also after second, mixture, fit to check that the discrepancy in the tails has improved.

        Note: assmes GEV, so that the exceedance probability above the quantile is modelled with Pareto distribution
    
        """

        #one distribution case
        if distrib2 == None:
            #Formulas to go from GEV parameters to Pareto parameters
            excess_scale =  distrib.kwds['scale'] - distrib.kwds['c'] * (distrib.ppf(q)- distrib.kwds['loc'])
            mean_above_q = distrib.ppf(q) + excess_scale/(1+distrib.kwds['c'])
            #TODO: check again formula to connect scale of Pareto to one of GEV to be sure

            mask_q = self.data_targ>distrib.ppf(q)

            #theoretical value
            mat_teo = np.mean((mean_above_q[mask_q] - distrib.ppf(q)[mask_q])/distrib.ppf(q)[mask_q]) 
            #empirical value
            mat_emp = np.mean((self.data_targ[mask_q] - distrib.ppf(q)[mask_q])/distrib.ppf(q)[mask_q])

        #mixture case (GEV specific)
        else:
            p0 = self.coefficients_fit[-1- self.p_time]
            if self.p_time == 1:
                p1 = self.coefficients_fit[-1]
            else:
                p1 = 0
            p = p0 + p1*self.data_pred['GMT_t']
            qr = q_arr #TODO: check that this is an array, not a single value
            den = p*(distrib.sf(qr)) + (1-p)*(distrib2.sf(qr))
            
            weight1 = p*(distrib.sf(qr))/den
            weight2 = (1-p)*(distrib2.sf(qr))/den
            
            if 'c' in self.expr_fit_mix.parameters_list:
                excess_scale1 =  distrib.kwds['scale'] - distrib.kwds['c'] * (distrib.ppf(q)- distrib.kwds['loc'])
                excess_scale2 =  distrib2.kwds['scale'] - distrib2.kwds['c'] * (distrib2.ppf(q)- distrib2.kwds['loc'])


                mean_above_q = qr + weight1*excess_scale1/(1+distrib.kwds['c']) + weight2*excess_scale2/(1+distrib2.kwds['c'])
            
            else:
                excess_scale1 =  distrib.kwds['scale'] - distrib.kwds['c'] * (distrib.ppf(q)- distrib.kwds['loc'])
                print('Normal Model for mixture not implemented yet')

            mask_q = self.data_targ>qr
            
            #theoretical value
            mat_teo = np.mean((mean_above_q[mask_q] - qr[mask_q])/qr[mask_q])
            #empirical value 
            mat_emp = np.mean((self.data_targ[mask_q] - qr[mask_q])/qr[mask_q]) 
            
        return mat_emp - mat_teo


    def stopping_rule(self, distrib):
        # evaluating threshold over time
        thres_t = distrib.isf(q=1 / self.threshold_stopping_rule)

        # selecting the minimum over the years to check
        thres = np.min(thres_t[self.ind_year_threshold])

        # identifying where exceedances occur
        if self.exclude_trigger:
            ind_data_stopped = self.data_targ > thres
        else:
            ind_data_stopped = self.data_targ >= thres

        # identifying remaining positions
        ind_data_ok = ~ind_data_stopped
        return ind_data_ok, ind_data_stopped

    def fullcond_thres(self, distrib):
        # calculating 2nd term for full conditional of the NLL
        # fc1 = distrib.logcdf(self.data_targ)
        fc2 = distrib.sf(self.data_targ)

        # return np.sum( (self.weights_driver * fc1)[self.ind_stopped_data] )
        # TODO: not 100% sure here, to double-check

        return np.log(np.sum((self.weights_driver * fc2)[self.ind_stopped_data]))

    def bic(self, coefficients):
        return len(coefficients) * np.log(self.n_sample) / self.n_sample - 2 * self.loglike(coefficients) 
        # TODO: remove /self.n_sample? bc weights are already normalized...
    
    def rectification_threshold(self, min_cdf_0 = 0.01):
        return self._get_threshold(min_cdf_0 = min_cdf_0)
    

    def mixture_cdf(self, distrib1, distrib2, p, x):
        #Computes CDF of mixture (2 components) with mixing factor p
        cdf1 = distrib1.cdf(x)
        cdf2 = distrib2.cdf(x)
        return p*cdf1 + (1-p)*cdf2

    def quantiles_func_mixture(self, coeffs1, coeffs2, p, qlist=[0.5, 0.05, 0.95, 0.005, 0.995]):
        """
        Function to compute quantiles of mixture distribution using interpolation of the inverse of the CDF
        (N.B. the CDF of the mixture is the mixture of the CDFs, but
        the quantile of the mixture is NOT the mixture of the quantiles)
        """
        x_grid = np.linspace(np.min(self.data_targ), np.max(self.data_targ), 1000)
        
        distrib1 = self.expr_fit.evaluate(coeffs1, self.data_pred)
        distrib2 = self.expr_fit_mix.evaluate(coeffs2, self.data_pred)
        cdf_grid = np.array([self.mixture_cdf(distrib1, distrib2, p, x) for x in x_grid])
        
        quantiles_dict = {}
        for q in qlist:
            quantiles_dict[str(q)] = np.zeros(cdf_grid.shape[1])

        #create interpolation of the inverse cdf to get quantile function
        #TODO: a bit slow, maybe better method
        for t in range(cdf_grid.shape[1]):  # Loop over each time step (each column)
            cdf_at_t = cdf_grid[:, t]  # Get the CDF values at this time step
            quantile_func = interp1d(cdf_at_t, x_grid, bounds_error=False, fill_value="extrapolate")
            for q in qlist:
                quantiles_dict[str(q)][t] = quantile_func(q)

        return quantiles_dict

    def crps(self, coeffs):
        # ps.crps_quadrature cannot be applied on conditional distributions, thu
        # calculating in each point of the sample, then averaging
        # NOTE: WARNING, TAKES A VERY LONG TIME TO COMPUTE

        tmp_cprs = []
        for i in np.arange(self.n_sample):
            distrib = self.expr_fit.evaluate(
                coeffs, {p: self.data_pred[p][i] for p in self.data_pred}
            )
            tmp_cprs.append(
                ps.crps_quadrature(
                    x=self.data_targ[i],
                    cdf_or_dist=distrib,
                    xmin=-10 * np.abs(self.data_targ[i]),
                    xmax=10 * np.abs(self.data_targ[i]),
                    tol=1.0e-4,
                )
            )

        # averaging
        return np.sum(self.weights_driver * np.array(tmp_cprs))

    def mean_fit(self, preds_np, mean_vals):
        """
        Function to compute mean of the distribution as a function of the predictor,
        useful to find scaling of extreme precipitation with GMT for example.
        Tries both linear and quadrati fits, returns coefficients of fit with smaller RMSE

        """
        log_filename = "process_log_err.log"
        with open(log_filename, "a") as f:
            #f.write(f"Gridpoint {gp} being processed by Process {pid}\n")
            if np.any(np.isnan(preds_np)) or np.any(np.isnan(mean_vals)):
                f.write(f"NaNs found in input arrays!")
                f.write(f"NaNs in preds_np: {np.sum(np.isnan(preds_np))}")
                f.write(f"NaNs in mean_vals: {np.sum(np.isnan(mean_vals))}, {self.coefficients_fit}")
        
        #Note: the mean is infinite if the shape is >=1 for a GEV
        # Remove NaN values (keeping corresponding x and y aligned)
        valid_mask = ~np.isnan(preds_np) & ~np.isnan(mean_vals)
        x = preds_np[valid_mask]
        y = mean_vals[valid_mask]
        
        # If no valid data remains, return NaNs
        if len(x) == 0 or len(y) == 0:
            return np.array([np.nan, np.nan, np.nan])  # Return NaN coefficients


        linear_coeffs = np.polyfit(x, y, 1)
        y_linear = np.polyval(linear_coeffs, x)

        quadratic_coeffs = np.polyfit(x, y, 2)
        y_quadratic = np.polyval(quadratic_coeffs, x)

        # Compute Log-Likelihoods (based on residuals)
        log_likelihood_linear = -0.5 * np.sum((y - y_linear)**2)
        log_likelihood_quadratic = -0.5 * np.sum((y - y_quadratic)**2)

        # Number of parameters
        k_linear = 2  # Slope + intercept
        k_quadratic = 3  # a, b, c

        # Compute AIC
        aic_linear = 2 * k_linear - 2 * log_likelihood_linear
        aic_quadratic = 2 * k_quadratic - 2 * log_likelihood_quadratic
        
        # RMSE for Linear Model
        rmse_linear = np.sqrt(mean_squared_error(y, y_linear))

        # RMSE for Quadratic Model
        rmse_quadratic = np.sqrt(mean_squared_error(y, y_quadratic))
        
        with open(log_filename, "a") as f:
            if np.isnan(rmse_linear) or np.isnan(rmse_quadratic):
                f.write("NaN encountered in RMSE values!")

        if rmse_linear < rmse_quadratic or np.abs(quadratic_coeffs[0] < 1e-3):
            return linear_coeffs #ax+b
        else:
            return quadratic_coeffs #ax**2 + bx + c

    @ignore_warnings  # suppress nan & inf warnings
    def fit(self):
        """
        Main fitting function, works with mixture and rectification.

        Currently starts with First Guess of mixture component; TODO: bring out first guess estimation
        """


        # ------FIRST GUESS MIXTURE CASE (Not as refined as one-distribution First Guess)
        #To improve stability, repeating first guess estimation multiple times and taking best option,
        # based on values of shape parameter that are more stable (between 0 and 0.5 for finite mean and variance)
        # Note: dependent on GEV
        #TODO: to be improved and generalized. First guess of mixture important for stability of fit
        if self.mix_init != 2:
            fct_mi = 1 #iterations in minimize function
            if self.first_guess is None:
                num_attempts_fg = 4
                best_coeffs = None

                for _ in range(num_attempts_fg):
                    self.find_fg()

                    #1st mixture component
                    coeffs = self.fg_coeffs.copy()

                    #shape parameter check for finite mean and variance
                    if coeffs[-1] <0 and coeffs[-1] >=-0.45:
                        best_coeffs = coeffs
                        break #stop if condition is satisfied, otherwise keep trying
                    if coeffs[-1]<-0.5:
                        coeffs[-1] = -0.45 #if shape still too large, I force a smaller value as first guess
                    best_coeffs = coeffs

                self.fg_coeffs = best_coeffs
                ffg_mix0 = np.copy(self.fg_coeffs) #1st mixture component
                ffg_mix = np.zeros(len(self.expr_fit_mix.coefficients_list)) #2nd mixture component, to be filled

                #location parameter of mixture component set to lower quantile of the smaller cluster of data
                ffg_mix[0] = np.quantile(self.data_targ[~self.mask_cluster], 0.05)

                #if time-dependent location, the varying coefficient initialized to same trend as main mix component
                if len(self.expr_fit_mix.coefficients_dict['loc'])>1:
                    ffg_mix[1] = ffg_mix0[1]
                
                #scale parameter first guess for mixture based on interquartile range/1.349 (valid more precisely for normally distributed data, but first guess)
                ncoeffs_scale = len(list(self.expr_fit_mix.coefficients_dict.values())[0]) #index of first coefficients in scale parameter, it starts after location
                ffg_mix[ncoeffs_scale] = (np.quantile(self.data_targ[~self.mask_cluster], 0.75) - np.quantile(self.data_targ[~self.mask_cluster], 0.25))/1.349 
                
                if len(self.expr_fit_mix.coefficients_dict['scale'])>1:
                    ffg_mix[ncoeffs_scale+1] = 0 #constant scale initial guess
                
                ### Focusing on smaller mixture component (second distribution)

                #assuming shape is the last param
                #shape initialized from skewness
                #TODO: more robust first guess?
                fg_skew = skew(self.data_targ[~self.mask_cluster])
                fg_shape = fg_skew/2/(1+fg_skew**2)

                ffg_mix[-1] = -fg_shape #c = - shape
                
                bounds = []
                for i in range(len(ffg_mix)):
                    bounds.append((-np.inf, np.inf))
                
                bounds[ncoeffs_scale] = (0, ffg_mix[0])
                bounds[-1] = (-0.5, 0.5)

                #bounds not used in the minimization in the end
                #minimization restricted to data in the second component mixture cluster
                #used as refined first guess of mixture coeffs
                m_mix = self.minimize(
                    func=self.fg_fun_NLL_mix,
                    x0=ffg_mix,
                    #bounds = bounds,
                    fact_maxfev_iter=1,
                    option_NelderMead="best_run",
                )

                #updated first guess for mixture
                fg_mix = m_mix.x
                distrib2 = self.expr_fit_mix.evaluate(fg_mix, self.data_pred)
                
                btm2 = distrib2.ppf(1e-6) #lower value, possible to use to check support
                
                #explicit expression of scale in case of time dependence
                if len(self.expr_fit_mix.coefficients_dict['scale'])>1:
                    fg_mix_scale_arr = fg_mix[ncoeffs_scale] + self.data_pred['GMT_t']*fg_mix[ncoeffs_scale+1]
                    fg_mix_scale_arr_mix = (fg_mix[ncoeffs_scale] - self.data_pred['GMT_t']*fg_mix[ncoeffs_scale+1])/fg_mix[ncoeffs_scale] #to check how close the scale gets to zero
                else:
                    fg_mix_scale_arr = fg_mix[ncoeffs_scale]
                    fg_mix_scale_arr_mix = 1
    

                #Checking that the scale stays positive across GMT values
                if fg_mix[ncoeffs_scale]<0 or np.any(fg_mix_scale_arr<0) or np.any(distrib2.cdf(0)>1e-5) or np.any(fg_mix_scale_arr_mix<0.1):

                    fg_mix[ncoeffs_scale] = ffg_mix[ncoeffs_scale] #keep previous value of first guess, not from minimization
                    if len(self.expr_fit_mix.coefficients_dict['scale'])>1:
                        fg_mix[ncoeffs_scale+1] = 0 #set to constant scale as first guess

                #As a first guess for the shape, I take it as small as possible, to avoid very large support
                #TODO: maybe in the future consider mix with normal distribution, one less parameter
                if np.abs(fg_mix[-1]) > 0.1:
                    fg_mix[-1] = -0.01

                #If shape from minimize is larger than initial estimate of shape, I take the latter
                #TODO: summarize these checks, some of them are superfluous
                if np.abs(fg_mix[-1])>0.8 or np.any(distrib2.cdf(0)>1e-5) or np.abs(fg_mix[-1])>np.abs(fg_shape):
                    fg_mix[-1] = -fg_shape
                #If the location parameter is negative it points to something wrong, maybe problem in clusters,
                #I take the first set of parameters before the minimize on the cluster
                if fg_mix[0]<0:
                    fg_mix = ffg_mix

                 
                #First guess for the mixing factor as (constant) ratio of the clusters
                fg_pmix = np.sum(self.mask_cluster)/len(self.data_targ)
                fg_pmix1 = 0

                ratio = []
                
                #Informing ratio parameter based on evolution of ratio of two clusters, 
                #if I have enough points in the predictors
                if np.any(self.data_pred['GMT_t']>1.5):
                    for n in np.arange(1, np.max(self.data_pred['GMT_t']), 0.5):
                        mask_gmt_all = np.sum((self.data_pred['GMT_t'] < n) & (self.data_pred['GMT_t'] >= n-0.5))
                        mask_gmt_cluster = np.sum((self.data_pred['GMT_t'][self.mask_cluster] <n) & (self.data_pred['GMT_t'][self.mask_cluster] >= n -0.5))


                        ratio.append([n, mask_gmt_cluster/mask_gmt_all])

                    ratio = np.array(ratio)
                    slope, intercept = np.polyfit(ratio[:,0], ratio[:,1], 1)

                    fg_pmix = np.min((intercept, 0.98))
                    fg_pmix1 = slope

                #in case of very few points in the smaller cluster, p is close to 1,
                #but I initialize it a bit further a way from 1 to detect the mixture
                if fg_pmix > 0.99 and fg_pmix<1:
                    fg_pmix = 0.97
                    fg_mix[-1] = -0.01


                ### Focusing on larger mixture component (first distribution)

                fg_skew = skew(self.data_targ[self.mask_cluster])
                fg_shape = fg_skew/2/(1+fg_skew**2)
                ffg_mix0[-1] = -fg_shape #I expect also the first GEV parameters to change
                ffg_mix0[ncoeffs_scale] = (np.quantile(self.data_targ[self.mask_cluster], 0.75) - np.quantile(self.data_targ[self.mask_cluster], 0.25))/1.349
                
                #As a first guess for shape I avoid very large values.
                if fg_mix[-1]<-0.25:
                    fg_mix[-1] = -0.25

                #no rectification applied when I do mixture
                res_rect = np.array([np.nan, np.nan])

                #Putting together the first guess parameters
                if self.p_time == 1: #time dependent mixture
                    fg_tot = np.concatenate((ffg_mix0, fg_mix, [fg_pmix], [fg_pmix1]))
                else:
                    fg_tot = np.concatenate((ffg_mix0, fg_mix, [fg_pmix]))

            #Case where first guess params are provided as input
            else:
                fg_tot = self.first_guess

        #---------------------
        # First guess standard case (with Rectification)
        else:
            fct_mi = 1 #iterations in the minimize function
            if self.func_first_guess is not None:
                self.func_first_guess()
            else:
                #self.find_fg()
                #sometimes first guess is unstable (produces high positive values of c, which does not make sense for precip)
                num_attempts_fg = 6
                best_coeffs = None

                for _ in range(num_attempts_fg):
                    self.find_fg()
                    coeffs = self.fg_coeffs.copy()

                    #shape grater than 0.5 leads to extremely high quantiles (infinite variance and higher moments)
                    #I iterate first guess until I find reasonable params
                    if coeffs[-1] <0 and coeffs[-1] >=-0.4: 
                        best_coeffs = coeffs
                        break
                    
                #In case of rectification, I base by shape estimate not on whole distribution (affected by discrete probability mass around 0),
                # but on the pdf above the rectification threshold (Pareto)
                if self.phys_thres > 0:
                    shape_gpd, _, _ = stats.genpareto.fit(self.data_targ[self.data_targ > self.phys_thres] - self.phys_thres)
                    coeffs[-1] = -shape_gpd

                #the idea is still to avoid exceptionally large supports when shape is too big
                #TODO: Generalize these checks, a bit too specific
                if coeffs[-1]<-0.4:
                    coeffs[-1] = -0.3 

                best_coeffs = coeffs
                self.fg_coeffs = best_coeffs
            fg_tot = self.fg_coeffs
            

        m = self.minimize(
            func=self.func_optim,
            x0=fg_tot,
            fact_maxfev_iter=fct_mi,
            option_NelderMead="best_run",
        )

        ###Fitting the rectified portion of data using a truncated normal to obtain piecewise continuous pdf,
        # instead of discrete + continuous
        #TODO: move this outside so I don't have to do it every iteration
        if self.phys_thres >=0: 
            pred = self.data_pred['GMT_t']
            if len(self.data_pred)>1:
                pred2_lab = [key for key in self.data_pred.keys() if key != 'GMT_t'][0]
                pred2 = self.data_pred[pred2_lab]
                distrib = self.expr_fit.evaluate(m.x, {'GMT_t': pred, pred2_lab: pred2})
            else:
                distrib = self.expr_fit.evaluate(m.x, {'GMT_t': pred})
            pdf_upper = np.mean(distrib.pdf(self.phys_thres))

            fg_rect = np.array([np.mean(self.data_targ[self.data_targ < self.phys_thres]), np.std(self.data_targ[self.data_targ < self.phys_thres])])

            def pdf_continuity_constraint(params):
                """Ensures f_lower(phys_thres) = f_upper(phys_thres)"""
                mu, sigma = params
                if sigma <= 0:
                    return np.inf  # Avoid invalid std values
    
                a_std = (0 - mu) / sigma
                b_std = (self.phys_thres - mu) / sigma
                trunc_norm = stats.truncnorm(a_std, b_std, loc=mu, scale=sigma)
    
                return trunc_norm.pdf(self.phys_thres) - pdf_upper

            constraint = {'type': 'eq', 'fun': pdf_continuity_constraint}
            
            #parameter estimation of the truncated normal with continuity constraint with the other part of the pdf
            m_rect = minimize(self.neg_loglike_rect, fg_rect, method="SLSQP", bounds=[(0, self.phys_thres), (1e-3, None)], constraints=constraint)
            res_rect = m_rect.x #rectification parameters

        else:
            #no rectification applied
            res_rect = np.array([np.nan, np.nan])

        # checking if the fit has failed
        if self.error_failedfit and not m.success:
            raise ValueError("Failed fit.")
        else:
            self.coefficients_fit = m.x
            self.coefficients_fit_rect =  res_rect
            self.eval_quality_fit()

    def eval_quality_fit(self):
        """Collecting multiple fit metrics"""

        # initialize
        self.quality_fit = {}

        # basic result: optimized value
        if "func_optim" in self.scores_fit:
            self.quality_fit["func_optim"] = self.func_optim(self.coefficients_fit)

        # The length of the coefficients informs the case I am in (Mixture, No mixture)
        coefficients = self.coefficients_fit

        if len(coefficients) == self.n_coeffs: #mixture case
            p0 = coefficients[-1 - self.p_time]
            if self.p_time == 1:
                p1 = coefficients[-1]
            else:
                p1 = 0
            p = p0 + p1 * self.data_pred['GMT_t']
            #probability of the mixture, self.mix_init is only the initial value for the minimization
            n = len(self.expr_fit.coefficients_list)
            coeffs1 = coefficients[:n]
            coeffs2 = coefficients[n:-1- self.p_time]

        else: #no mixture
            coeffs1 = coefficients

        #If shape is unreasonably large (more than 1)
        if coeffs1[-1]<=-1:
            log_filename = "process_log_err.log"
            with open(log_filename, "a") as f:
                f.write('c less than -1, adjusted to -0.98, check func_optim, fit might have failed\n')
                coeffs1[-1] = -0.98

        #compute loglikelihood for interior points
        distrib = self.expr_fit.evaluate(coeffs1, self.data_pred)

        if len(coefficients) == self.n_coeffs: #mixture case #TODO: maybe bring in in previous if
            distrib = self.expr_fit.evaluate(coeffs1, self.data_pred)
            distrib2 = self.expr_fit_mix.evaluate(coeffs2, self.data_pred)
            quantiles_dict = self.quantiles_func_mixture(coeffs1, coeffs2, p, qlist=[0.5, 0.05, 0.95, 0.005, 0.995])
            q95 = quantiles_dict['0.95']
            q995 = quantiles_dict['0.995']
            q50 = quantiles_dict['0.5']
            q05 = quantiles_dict['0.05']
        else:
            distrib2 = None
                
        # NLL averaged over sample
        if "NLL" in self.scores_fit:
            self.quality_fit["NLL"] = self.neg_loglike(self.coefficients_fit)

        # BIC averaged over sample
        if "BIC" in self.scores_fit:
            self.quality_fit["BIC"] = self.bic(self.coefficients_fit)

        # CRPS
        if "CRPS" in self.scores_fit:
            self.quality_fit["CRPS"] = self.crps(self.coefficients_fit)
        
        # rectification threshold
        if "rectification" in self.scores_fit:
            self.quality_fit["rectification"] = self.phys_thres
            self.quality_fit["rect_loc"] = self.coefficients_fit_rect[0] #location of truncated normal
            self.quality_fit["rect_std"] = self.coefficients_fit_rect[1] #std dev of truncated normal

        #counting the points above percentiles, to compare them with theoretical values
        if len(coefficients) == self.n_coeffs: #mixture case
            if "count_95" in self.scores_fit:
                count = np.sum(self.data_targ>q95)/len(self.data_targ)
                self.quality_fit["count_95"] = count
            if "count_995" in self.scores_fit:
                count = np.sum(self.data_targ>q995)/len(self.data_targ)
                self.quality_fit["count_995"] = count
            if "count_50" in self.scores_fit:
                count = np.sum(self.data_targ>q50)/len(self.data_targ)
                self.quality_fit["count_50"] = count
            if "count_05" in self.scores_fit:
                count = np.sum(self.data_targ>q05)/len(self.data_targ)
                self.quality_fit["count_05"] = count

        else: #no mixture case
            if "count_95" in self.scores_fit:
                count = np.sum(self.data_targ>distrib.ppf(0.95))/len(self.data_targ)
                self.quality_fit["count_95"] = count
            if "count_995" in self.scores_fit:
                count = np.sum(self.data_targ>distrib.ppf(0.995))/len(self.data_targ)
                self.quality_fit["count_995"] = count
            #the quantiles change in case of rectification, 
            #so I need to account for the piecewise pdf in the lower quantiles
            if "count_50" in self.scores_fit:
                coeffs_rect = self.coefficients_fit_rect
                if self.phys_thres >= 0 and coeffs_rect[1] >0 and np.mean(distrib.median()) < self.phys_thres:
                    a = self.phys_thres
                    lb = 0
                    distrib_rect = truncnorm((lb - coeffs_rect[0])/coeffs_rect[1], (a - coeffs_rect[0])/coeffs_rect[1], loc = coeffs_rect[0], scale = coeffs_rect[1])
                    q50 = np.where(distrib.cdf(a)>= 0.5, distrib_rect.ppf(0.5/distrib.cdf(a)), distrib.ppf(0.5))
                    count = np.sum(self.data_targ>q50)/len(self.data_targ)
                else:
                    count = np.sum(self.data_targ>distrib.ppf(0.50))/len(self.data_targ)
                self.quality_fit["count_50"] = count
            if "count_05" in self.scores_fit:
                coeffs_rect = self.coefficients_fit_rect
                if self.phys_thres >= 0 and coeffs_rect[1] >0 and np.mean(distrib.ppf(0.05)) < self.phys_thres:
                    a = self.phys_thres
                    lb = 0
                    distrib_rect = truncnorm((lb - coeffs_rect[0])/coeffs_rect[1], (a - coeffs_rect[0])/coeffs_rect[1], loc = coeffs_rect[0], scale = coeffs_rect[1])
                    q05 = np.where(distrib.cdf(a)>= 0.05, distrib_rect.ppf(0.05/distrib.cdf(a)), distrib.ppf(0.05))
                    count  = np.sum(self.data_targ>q05)/len(self.data_targ)
                else:
                    count = np.sum(self.data_targ>distrib.ppf(0.05))/len(self.data_targ)
                self.quality_fit["count_05"] = count
            

        if "mean_above_threshold_95" in self.scores_fit:
            if len(coefficients) == self.n_coeffs:
                qq = q95
            else:
                qq = None
            self.quality_fit["mean_above_threshold_95"] = self.mean_above_threshold(distrib, distrib2, q=0.95, q_arr = qq)
            
            #TODO: update computation using the mixtureof distributions
        if "mean_above_threshold_995" in self.scores_fit:
            if len(coefficients) == self.n_coeffs:
                qq2 = q995
            else:
                qq2 = None
            self.quality_fit["mean_above_threshold_995"] = self.mean_above_threshold(distrib, distrib2, q=0.995, q_arr = qq2)

        # silhouette parameter
        if "silhouette" in self.scores_fit:
            self.quality_fit["silhouette"] = self._silhouette_score()

        # spread of the clusters
        if "spread_clusters" in self.scores_fit:
            self.quality_fit["spread_clusters"] = self.spread_clusters

        # size of the small cluster
        if "small_cluster" in self.scores_fit:
            self.quality_fit["small_cluster"] = self.nsmall_cluster

        # Distribution Mean vs GMT
        if "scaling_mean" in self.scores_fit:
            if False:#np.isinf(self.func_optim(self.coefficients_fit)):
                self.quality_fit["scaling_mean"] = str(np.nan) # I want to have a reference value anyway, but check func_optim
            else:
                #rectified case
                if self.phys_thres >= 0:
                    coeffs_rect = self.coefficients_fit_rect
                    a = self.phys_thres

                    #If I use truncated normal
                    if coeffs_rect[1] >0:
                        lb = 0
                        distrib_rect = truncnorm((lb - coeffs_rect[0])/coeffs_rect[1], (a - coeffs_rect[0])/coeffs_rect[1], loc = coeffs_rect[0], scale = coeffs_rect[1])
                        part1 = distrib.cdf(a) * distrib_rect.mean()
                    #If I use just rectification threshold a
                    else:
                        part1 = a * distrib.cdf(a) #this has length of predictors
                    print('mean for rectification being computed')
                    integs = []
                    xvals = []
                    #TODO: add other covariates to see the evolution of mean

                    #TODO: Take less points here, because in the end I am going to fit the mean, so I dont need 4144 points...
                    for t in range(0, len(self.data_targ), 4):
                        integrand = lambda x: x * distrib.pdf(x)[t]
                        integral, error = quad(integrand, a, np.inf)
                        integs.append(integral + part1[t])
                        xvals.append(self.data_pred['GMT_t'][t])

                    xvals = np.array(xvals)
                    integs = np.array(integs)


                    if np.sum(np.isnan(integs))>0 or np.sum(np.isnan(xvals))>0:
                        print('The rectified integral for the mean has NaN values', np.sum(np.isnan(integs)))

                    self.quality_fit["scaling_mean"] = str(self.mean_fit(xvals, integs))
                
                else:
                    #standard case
                    mean_vals = distrib.mean()
                    if np.sum(np.isnan(mean_vals))>0 or np.sum(np.isnan(self.data_pred['GMT_t']))>0:
                        print('The standard computation of distrib mean has nan values: ', np.sum(np.isnan(mean_vals)), self.coefficients_fit)
                    coeffs = self.mean_fit(self.data_pred['GMT_t'],mean_vals)
            
                    #mixture case
                    if distrib2 != None:
                        mean_vals2 = distrib2.mean()
                        if np.sum(np.isnan(mean_vals2))>0:
                            print('mean values of 2nd distribution has Nan values: gridpoint')
                        coeffs2 = self.mean_fit(self.data_pred['GMT_t'], mean_vals2)
                        coeffs = np.concatenate((coeffs, coeffs2, [p0, p1]))
                    self.quality_fit["scaling_mean"] = str(coeffs)

