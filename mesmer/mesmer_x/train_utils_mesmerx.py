# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Refactored code for the training of distributions

"""

import math

import numpy as np
import scipy as sp
import xarray as xr

#Added for Rx1day application
from scipy.stats import truncnorm

_DISCRETE_DISTRIBUTIONS = sp.stats._discrete_distns._distn_names
_CONTINUOUS_DISTRIBUTIONS = sp.stats._continuous_distns._distn_names

_ALL_DISTRIBUTIONS = _DISCRETE_DISTRIBUTIONS + _CONTINUOUS_DISTRIBUTIONS

_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class Expression:

    def __init__(self, expr, expr_name):
        """conditional distribution: definition, interpretation and evaluation

        When initialized, the class identifies the distribution, inputs, parameters and
        coefficients. Then, the next function would be evaluate(coefficients, inputs,
        forced_shape) to assess the distribution given the provided values of
        coefficients & inputs. NB language: the distribution depends on parameters
        (loc, scale, etc), that are functions of inputs and coefficients.

        Parameters
        ----------
        expr : str
            string describing the expression that will be used. Existing methods for
            flexible conditional distributions don't provide the level of flexibility
            that MESMER-X uses.
            Requirements to follow to have the expression being understood:

            - distribution: must be the name of a distribution in scipy stats (discrete
              or continuous): https://docs.scipy.org/doc/scipy/reference/stats.html

            - parameters: all names for the parameters of the distributions must be
              provided: loc, scale, shape, mu, a, b, c, etc.

            - coefficients: must be named "c#", with # being the number of the
              coefficient: e.g. c1, c2, c3.

            - inputs: any string that can be written as a variable in python, and
              surrounded with "__": e.g. __GMT__, __X1__, __gmt_tm1__, but NOT
              __gmt-1__, __#days__, __GMT, _GMT_ etc.

            - equations: the equations for the evolutions must be written as it would be
              normally. Names of packages should be included.
              spaces dont matter in the equations

        expr_name : str
            Name of the expression.

        Notes
        -----
        - Forcing values of certain parameters (eg scale) to be positive: to implement
          in class 'expression'?
        - Forcing values of certain parameters (eg mu for poisson) to be integers: to
          implement in class 'expression'?
        - Reasons for not using sympy:
          - sympy.stats does not have all required distributions (e.g. GEV) & all
            required functions (e.g. log-likelihood)
            -> so using scipy distributions with parameters as sympy expressions
          - but these expressions dont deal well with numpy array, or even less with
            xarray -> could, but tedious.
          - And the result would be significantly slower than with numpy/xarray based
            approaches like here.

        Examples
        --------
        - "genextreme(loc=c1 + c2 * __pred1__, scale=c3 + c4 * __pred2__**2, c=c5)"
        - "norm(loc=c1 + (c2 - c1) / ( 1 + np.exp(c3 * __GMT_t__ + c4 * __GMT_tm1__ - c5) ), scale=c6)"
        - "exponpow(loc=c1, scale=c2+np.min([np.max(np.mean([__GMT_tm1__,__GMT_tp1__],axis=0)), math.gamma(__XYZ__)]), b=c3)"
        """

        # basic initialization
        self.expression = expr
        self.expression_name = expr_name

        # identify distribution
        self._interpret_distrib()

        # identify parameters
        self._find_parameters_list()
        self._find_expr_parameters()

        # identify coefficients
        self._find_coefficients()

        # identify inputs
        self._find_inputs()

        # correct expressions of parameters
        self._correct_expr_parameters()

    def _interpret_distrib(self):
        """interpreting the expression"""

        dist = str.split(self.expression, "(")[0]

        if dist not in _ALL_DISTRIBUTIONS:
            raise AttributeError(
                f"Could not find distribution '{dist}'."
                " Please provide a distribution written as in scipy.stats:"
                " https://docs.scipy.org/doc/scipy/reference/stats.html"
            )

        self.distrib = getattr(sp.stats, dist)
        self.is_distrib_discrete = dist in _DISCRETE_DISTRIBUTIONS

    def _find_expr_parameters(self):

        # removing spaces that would hinder the identification
        tmp_expression = self.expression.replace(" ", "")

        # removing distribution part
        tmp_expression = "(".join(str.split(tmp_expression, "(")[1:])
        tmp_expression = ")".join(str.split(tmp_expression, ")")[:-1])

        # identifying groups
        sub_expressions = str.split(tmp_expression, ",")

        self.parameters_expressions = {}
        for sub_exp in sub_expressions:
            param, sub = str.split(sub_exp, "=")
            if param in self.parameters_list:
                self.parameters_expressions[param] = sub
            else:
                raise ValueError(
                    f"The parameter '{param}' is not part of prepared expression in scipy.stats:"
                    " https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats."
                    f" of the distribution '{self.distrib.name}'"
                )

        # recommend not to try to fill in missing information on parameters with
        # constant parameters, but to raise a ValueError instead.
        for pot_param in self.parameters_list:
            if pot_param not in self.parameters_expressions.keys():
                raise ValueError(f"No information provided for `{pot_param}`")

    def _find_parameters_list(self):
        """
        List parameters for scipy.stats.distribution.
            distribution: a string or scipy.stats distribution object.

        Returns
            A list of distribution parameter strings.

        Based on the code available at: https://github.com/scipy/scipy/issues/9575
        """

        self.parameters_list = []

        if self.distrib.shapes:
            self.parameters_list += [
                name.strip() for name in self.distrib.shapes.split(",")
            ]

        if self.distrib.name in _DISCRETE_DISTRIBUTIONS:
            self.parameters_list += ["loc"]
        elif self.distrib.name in _CONTINUOUS_DISTRIBUTIONS:
            self.parameters_list += ["loc", "scale"]

        # prepary basic boundaries on parameters: incomplete, did not find a way to
        # evaluate automatically the limits on shape parameters

        self.boundaries_parameters = {
            p: [-np.inf, np.inf] for p in self.parameters_list
        }

        # scale must be positive
        if "scale" in self.boundaries_parameters:
            self.boundaries_parameters["scale"][0] = 0

    def _find_coefficients(self):
        """
        coefficients are supposed to be written as "c#", with "#" being a number.
        """

        self.coefficients_list, self.coefficients_dict = [], {}

        for param in self.parameters_expressions:
            self.coefficients_dict[param] = []
            # iniatilize detection
            cf = ""

            # adding one space at the end to append the last coefficient
            for pep in self.parameters_expressions[param] + " ":

                if pep == "c":
                    # starting expression for a coefficient
                    cf = "c"
                elif (cf == "c") and (pep in _DIGITS):
                    # continuing expression for a coefficient
                    cf += pep
                else:
                    if cf not in ["", "c"]:

                        # ending expression for a coefficient
                        if cf not in self.coefficients_list:
                            self.coefficients_list.append(cf)
                            self.coefficients_dict[param].append(cf)

                        cf = ""
                    else:
                        # not a coefficient, move to the next
                        pass

    def _find_inputs(self):

        self.inputs_list = []

        for param in self.parameters_expressions:
            terms = str.split(self.parameters_expressions[param], "__")
            for i in np.array(terms)[np.arange(1, len(terms), 2)]:
                if i not in self.inputs_list:
                    self.inputs_list.append(i)

        # require a specific order for use in correct_expr_parameters
        self.inputs_list.sort(key=len, reverse=True)

    def _correct_expr_parameters(self):
        # list of inputs and coefficients, sorted by descending length, to be sure that
        # when removing them, will remove the good ones and not those with their short
        # name contained in the long name of another

        tmp_list = self.inputs_list + self.coefficients_list + ["__"]
        tmp_list.sort(key=len, reverse=True)

        # making sure that the expression will be understood
        for param in self.parameters_expressions:

            # 1. checking the names of packages
            expr = self.parameters_expressions[param]

            # removing inputs and coefficients
            for x in tmp_list:
                expr = expr.replace(x, "")

            # reading this condensed expression to find terms to replace
            t = ""  # initialization

            # adding one space at the end to treat the last term
            for ex in expr + " ":

                # TODO: could do this using regular expressions
                if ex in ["(", ")", "[", "]", "+", "-", "*", "/", "%"] + _DIGITS:
                    # this is a readable character that will not be an issue to read
                    if t not in ["", " "]:
                        # a term is here, and needs to be confirmed as readable
                        if t.startswith("np."):
                            if t[len("np.") :] not in vars(np):
                                raise ValueError(
                                    f"Proposed a numpy function that does not exist: '{t}'"
                                )
                            else:
                                # nothing to replace, can go with that
                                pass
                        elif t.startswith("math."):
                            if t[len("math.") :] not in vars(math):
                                raise ValueError(
                                    f"Proposed a math function that does not exist: '{t}'"
                                )
                            else:
                                # nothing to replace, can go with that
                                pass
                        else:
                            raise ValueError(
                                f"Unknown function '{t}' in expression"
                                f" '{self.parameters_expressions[param]}' for '{param}'."
                                "Do you need to prepend it with 'np.' (or 'math.')?"
                                " Currently only numpy and math functions are supported."
                            )
                    else:
                        # was a readable character, is still a readable character,
                        # nothing to do
                        pass
                else:
                    t += ex

                # TODO: would make sense to invert the logic here - move building 't'
                # above the validity check

            # replacing names of inputs in expressions
            for i in self.inputs_list:
                self.parameters_expressions[param] = self.parameters_expressions[
                    param
                ].replace(f"__{i}__", i)

    def evaluate_params(self, coefficients_values, inputs_values, forced_shape=None):
        """
        Evaluates the parameters for the provided inputs and coefficients

        Parameters
        ----------
        coefficients_values : dict | xr.Dataset(c_i) | list of values
            Coefficient arrays or scalars. Can have the following form
            - dict(c_i = values or np.array())
            - xr.Dataset(c_i)
            - list of values
        inputs_values : dict | xr.Dataset
            Input arrays or scalars. Can be passed as
            - dict(inp_i = values or np.array())
            - xr.Dataset(inp_i)
        forced_shape : None | tuple or list of dimensions
            coefficients_values and inputs_values for transposition of the shape

        Returns
        -------
        params: dict
            Realized parameters for the given expression, coefficients and covariates;
            to pass ``self.distrib(**params)`` or it's methods.

        Warnings
        --------
        with xarrays for coefficients_values and inputs_values, the outputs with have
        for shape first the one of the coefficient, then the one of the inputs
        --> trying to avoid this issue with 'forced_shape'
        """

        # TODO:
        # - use broadcasting
        # - can we avoid using exec & eval?
        # - only parse the values once? (to avoid doing it repeatedly)
        # - require list of coefficients_values (similar to minimize)?
        # - convert dataset to numpy arrays?

        # Check 1: are all the coefficients provided?
        if isinstance(coefficients_values, dict | xr.Dataset):
            # case where provide explicit information on coefficients_values
            for c in self.coefficients_list:
                if c not in coefficients_values:
                    raise ValueError(f"Missing information for the coefficient: '{c}'")
        else:
            # case where a vector is provided, used for the optimization performed
            # during the training
            if len(coefficients_values) != len(self.coefficients_list):
                raise ValueError("Inconsistent information for the coefficients_values")

            coefficients_values = {
                c: coefficients_values[i] for i, c in enumerate(self.coefficients_list)
            }

        # Check 2: are all the inputs provided?
        for i in self.inputs_list:
            if i not in inputs_values:
                raise ValueError(f"Missing information for the input: '{i}'")

        # Check 3: do the inputs have the same shape
        shapes = {inputs_values[i].shape for i in self.inputs_list}
        if len(shapes) > 1:
            raise ValueError("shapes of inputs must be equal")

        # gather coefficients and covariates (can't use d1 | d2, does not work for dataset)
        locals = {**coefficients_values, **inputs_values}

        # evaluate parameters
        parameters_values = {}
        for param, expr in self.parameters_expressions.items():
            # may need to silence warnings here, to avoid spamming
            parameters_values[param] = eval(expr, None, locals)

        # Correcting shapes 1: scalar parameters must have the shape of the inputs
        if len(self.inputs_list) > 0:

            for param in self.parameters_list:
                param_value = parameters_values[param]

                # TODO: use np.ndim(param_value) ==  0? (i.e. isscalar)
                if isinstance(param_value, int | float) or param_value.ndim == 0:
                    if isinstance(coefficients_values, xr.Dataset) and (
                        isinstance(inputs_values, xr.Dataset)
                    ):
                        param_value = param_value * xr.ones_like(
                            inputs_values[self.inputs_list[0]]
                        )
                    else:
                        param_value = param_value * np.ones(
                            inputs_values[self.inputs_list[0]].shape
                        )

                parameters_values[param] = param_value

        # Correcting shapes 2: possibly forcing shape
        if len(self.inputs_list) > 0 and forced_shape is not None:

            for param in self.parameters_list:
                dims_param = [
                    d for d in forced_shape if d in parameters_values[param].dims
                ]
                parameters_values[param] = parameters_values[param].transpose(
                    *dims_param
                )

        return parameters_values

    def evaluate(self, coefficients_values, inputs_values, forced_shape=None):
        """
        Evaluates the distribution with the provided inputs and coefficients

        Parameters
        ----------
        coefficients_values : dict | xr.Dataset(c_i) | list of values
            Coefficient arrays or scalars. Can have the following form
            - dict(c_i = values or np.array())
            - xr.Dataset(c_i)
            - list of values
        inputs_values : dict | xr.Dataset
            Input arrays or scalars. Can be passed as
            - dict(inp_i = values or np.array())
            - xr.Dataset(inp_i)
        forced_shape : None | tuple or list of dimensions
            coefficients_values and inputs_values for transposition of the shape

        Returns
        -------
        distr: scipy stats frozen distribution
            Frozen distribution with the realized parameters applied to.

        Warnings
        --------
        with xarrays for coefficients_values and inputs_values, the outputs with have
        for shape first the one of the coefficient, then the one of the inputs
        --> trying to avoid this issue with 'forced_shape'
        """

        params = self.evaluate_params(coefficients_values, inputs_values, forced_shape)
        return self.distrib(**params)



def probability_integral_transform(
    data,
    target_name,
    expr_start,
    expr_end,
    expr_start_mix = None,
    expr_end_mix = None,
    coeffs_start=None,
    coeffs_start_mix = None,
    qual_start = None,
    qual_end = None,
    preds_start=None,
    coeffs_end=None,
    coeffs_end_mix = None,
    preds_end=None,
):
    """
    Probability integral transform of the data given parameters of a given distribution
    into their equivalent in a standard normal distribution.

    Parameters
    ----------
    data : not sure yet what data format will be used at the end.
        Assumed to be a xarray Dataset with coordinates 'time' and 'gridpoint', and one
        2D variable with both coordinates
    target_name : str
        name of the variable to train
    expr_start : str
        string describing the starting expression
    expr_end : str
        string describing the starting expression
    expr_start_mix : str or None
        string describing the starting expression for the mixture component, if present
    expr_end_mix : str or None
        string describing the starting expression for the mixture component, if present
    preds_start : not sure yet what data format will be used at the end.
        Covariants of the starting expression. Default: empty Dataset.
    coeffs_start : xarray dataset
        Coefficients of the starting expression. Default: empty Dataset.
    coeffs_start_mix : xarray dataset
        Coefficients of the starting expression for the mixture component. Default: empty Dataset.
    preds_end : not sure yet what data format will be used at the end.
        Covariants of the ending expression. Default: empty Dataset.
    coeffs_end : xarray dataset
        Coefficients of the ending expression. Default: empty Dataset.
    coeffs_end_mix : xarray dataset
        Coefficients of the ending expression for the mixture component. Default: empty Dataset.

    Returns
    -------
    transf_inputs : not sure yet what data format will be used at the end.
        Assumed to be a xarray Dataset with coordinates 'time' and 'gridpoint' and one
        2D variable with both coordinates

    Notes
    -----
    Assumptions:
    - Context: The transformation may fail if the values are very unlikely, leading to a
      CDF equal or too close from 0 or 1, which will raise issues.
    - Current solution: During training, this problem is avoided thanks to the option
      'threshold_min_proba', meaning that all points in sample would have a minimum
      probability.
    - Limit to solution: However, if transforming a sample not used during sample, and
      in a domain not represented in the training sample, it may cause issues.
    - Additional fix: If this situation is encountered, I suggest to block the CDF
      values 'cdf_item' within a domain: values with a probability of 1.e-99 could be
      forced to 1.e-9. Unless someone has a better idea! :D
    Disclaimer:
    - TODO
    can only take 2D input aka only one realisation for the back-transformation

    """
    # preparation of distributions
    expression_start = Expression(expr_start, "start")
    #mixture part if present
    if expr_start_mix is not None:
        expression_start_mix = Expression(expr_start_mix, "start")

    expression_end = Expression(expr_end, "end")
    if expr_end_mix is not None:
        expression_end_mix = Expression(expr_end_mix, "start")

    if coeffs_start is None:
        coeffs_start = xr.Dataset()

    if coeffs_end is None:
        flag_gev = False
        coeffs_end = xr.Dataset()
    else:
        flag_gev = True

    #Identify whether rectification is present in the initial data
    if qual_start is not None and 'rectification' in list(qual_start.data_vars):
        rect = qual_start.rectification
        rect_loc = qual_start.rect_loc #loc of truncated normal describing rectified portion
        rect_std = qual_start.rect_std #std of truncated normal describing rectified portion
        #if mixture is done instead of rectification i remove the mask for rectification in those points
        if expr_start_mix is not None: 
            cond_rect_up = (qual_start.rectification>=0) & (qual_start.rectification_mix<0)
            rect = xr.where(cond_rect_up, qual_start.rectification_mix, qual_start.rectification)
    else:
        rect = -np.inf

    if qual_end is not None and 'rectification' in list(qual_end.data_vars):
        rect_end = qual_end.rectification
        rect_loc_end = qual_end.rect_loc
        rect_std_end = qual_end.rect_std

        if expr_end_mix is not None:
            cond_rect_up = (qual_end.rectification>=0) & (qual_end.rectification_mix<0)
            rect_end = xr.where(cond_rect_up, qual_end.rectification_mix, qual_end.rectification)
    else:
        rect_end = -np.inf
    # transformation
    out = []

    # loop to change with new data structure of MESMER
    for i, item in enumerate(data):
        data_item, scen = item

        #identify values above rectification threshold
        mask_rect = data_item[target_name] >= rect
        
        if preds_start is None:
            preds_start_item = xr.Dataset()
        else:
            preds_start_item = preds_start[i][0]

        if preds_end is None:
            preds_end_item = xr.Dataset()
        else:
            preds_end_item = preds_end[i][0]

        print(f"Transforming {target_name}: {scen}")
        #I generate artificially the datapoints below the rectification threshold
        

        ###START

        # calculation distributions for this scenario
        distrib_start = expression_start.evaluate(
            coeffs_start, preds_start_item, forced_shape=data_item[target_name].dims
        )
        

        #First I compute the cdf only above rectification threshold
        #Handling carefully points with CDF too close to 0 or 1
        cdf_item = distrib_start.cdf(data_item[target_name].where(mask_rect))
        print('NaN values in CDF at start: ', np.sum(np.isnan(cdf_item)))

        if np.any(np.isnan(distrib_start.cdf(data_item[target_name]))):
            nan_coords = np.where(np.isnan(distrib_start.cdf(data_item[target_name])))
            cdf_item = np.where(np.isnan(distrib_start.cdf(data_item[target_name])), 0.5, cdf_item)
            print('gridpoints with ', len(nan_coords[-1]), ' nan values: ', set(nan_coords[-1]))

        print('One values in CDF start: ', np.sum(cdf_item == 1))
        print('Values above 1 in CDF at start: ', np.sum(cdf_item > 1))
        #To avoid explosions in values after transformation, I set thresholds (min probability)
        cdf_item = np.where(cdf_item == 1, 0.9999, cdf_item)
        cdf_item = np.where(cdf_item == 0, 1-0.9999, cdf_item)
        print('NaN values in CDF at start (2nd): ', np.sum(np.isnan(cdf_item)))

        if np.isscalar(rect)==False: #Rectification case
            thresholds = np.where(np.isinf(rect.values), np.nan, rect.values)
            lb = 1e-5 * np.ones_like(rect.values)
            ub = thresholds

            mus = rect_loc.values
            stds = rect_std.values

            lb_std = (lb - mus)/stds
            ub_std = (ub - mus)/stds
            

            distrib_rect = truncnorm(lb_std, ub_std, loc = mus, scale = stds)
            #The CDF of the truncated norm is 1 in the trheshold by default, so I need to match it with cdf(threshold)
            rect_cdf_vals = distrib_start.cdf(thresholds) * distrib_rect.cdf(data_item[target_name].values)
            rect_cdf_item = np.where(np.isnan(cdf_item), rect_cdf_vals, cdf_item)

            cdf_item = rect_cdf_item
            print('Nan values after rect: ', np.sum(np.isnan(cdf_item)))


        if coeffs_start_mix is not None: #mixture case
            distrib_start_mix = expression_start_mix.evaluate(
                coeffs_start_mix, preds_start_item, forced_shape=data_item[target_name].dims
            )

            cdf_item_mix = distrib_start_mix.cdf(data_item[target_name].where(mask_rect)) #TODO: check that mixture case doesnt overlap with rectification
            #Might get some warning because some points lie outside of the support of the second distribution,
            #But this would return a CDF of 0 or 1 regardless
            p0 = coeffs_start_mix.p0
            p1 = coeffs_start_mix.p1

            #CDF of mixture expression
            cdf_item_mix2 = (preds_start_item['GMT_t'] * p1 + p0).values * cdf_item + (- preds_start_item['GMT_t'] * p1 + 1 - p0).values * cdf_item_mix
            #print('Zero values in CDF mix: ', np.sum(cdf_item_mix2 == 0))
            #print('One values in CDF mix: ', np.sum(cdf_item_mix2 == 1))
            print('>1 values in CDF mix: ', np.sum(cdf_item_mix2>1), np.nanmax(cdf_item_mix2))
            cdf_item = np.where(np.isnan(p0.values), cdf_item, cdf_item_mix2) 
        
        #checks, can be removed
        print('Zero values in CDF: ', np.sum(cdf_item == 0))
        print('One values in CDF: ', np.sum(cdf_item == 1))
        print('Values above 1 in CDF: ', np.sum(cdf_item > 1))
        print('NaN values in CDF after Mix: ', np.sum(np.isnan(cdf_item)))
        print('Values below 0 in CDF: ', np.sum(cdf_item < 0))
        print('Min, Max: ', np.nanmin(cdf_item), np.nanmax(cdf_item))
        cdf_item = np.where(cdf_item == 1, 0.9999, cdf_item)
        cdf_item = np.where(cdf_item == 0, 1-0.9999, cdf_item)
        if flag_gev:
            cdf_item = np.where(cdf_item <= 1e-8, 1e-8, cdf_item)
            
            #I exclude events rarer than 1 in 2000 years
            print('Vals_abobe 1-5e-4:', np.sum(cdf_item >1-5e-4))
            cdf_item = np.where(cdf_item >1-5e-4, 1-5e-4, cdf_item)
            print('Vals_above 1-5e-4 afterwards:', np.sum(cdf_item >1-5e-4))



        ###END
        distrib_end = expression_end.evaluate(
            coeffs_end, preds_end_item, forced_shape=data_item[target_name].dims
        )
        
        # corresponding values on the ending distribution
        transf_item = distrib_end.ppf(cdf_item)
        print('INF values in PPF: ', np.sum(np.isinf(transf_item))/len(transf_item))
        print('NaN values in PPF: ', np.sum(np.isnan(transf_item))/len(transf_item))

        #if coeffs_end_mix is not None:
        #Ways to estimate PPF of MIXTURE distribution
        if False: #(skipped, takes too long)
            distrib_end_mix = expression_end_mix.evaluate(
                coeffs_end_mix, preds_end_item, forced_shape=data_item[target_name].dims
            )

            p0 = coeffs_end_mix.p0
            p1 = coeffs_end_mix.p1
            p = (preds_end_item['GMT_t'] * p1 + p0).values
            
            #since in the mixture usually p is close to 1 and 1-p close to zero, 
            #I take as a first guess the sum of the inverses of the individual cdfs
            xg = p * distrib_end.ppf(cdf_item) + (1-p) * distrib_end_mix.ppf(cdf_item)

            tol = 1e-3
            max_steps = 200

            for i in range(max_steps):
                correction = (p * distrib_end.cdf(xg) + (1-p) * distrib_end_mix.cdf(xg)) - cdf_item
                xg = xg - correction #I correct the initial guess based on the distances between cdf values

                error = np.abs(correction)/np.abs(xg)

                if np.nanmax(error) < tol:
                    break
            if np.any(error > 1e-2):
                print('Final maximum error in the transformed values larger than 1%')
            
            print('Final error on ppf of mixture :', np.nanmax(error))
            transf_item_mix = xg

            transf_item_tot = np.where(np.isnan(transf_item_mix), transf_item, transf_item_mix)
            transf_item = transf_item_tot

        #alternative faster method to estimate Mixture PPF based on bisection
        if coeffs_end_mix is not None: 
            distrib_end_mix = expression_end_mix.evaluate(
                coeffs_end_mix, preds_end_item, forced_shape=data_item[target_name].dims
            )

            p0 = coeffs_end_mix.p0
            p1 = coeffs_end_mix.p1
            p = (preds_end_item['GMT_t'] * p1 + p0).values
            
            xg = p * distrib_end.ppf(cdf_item) + (1-p) * distrib_end_mix.ppf(cdf_item)
            
            x_L = np.full_like(p, 0)  # Lower bound
            x_U = np.full_like(p, 3*np.nanmax(xg))  # Upper bound

            tol = 1e-5  #Note: the cdf is non linear, so even a small difference close to cdf = 1 changes a lot in the ppf...
            max_iters = 100

            for i in range(max_iters):
                x_M = (x_L + x_U)/2
                F_M = p* distrib_end.cdf(x_M) + (1-p) * distrib_end_mix.cdf(x_M)

                mask_lo = F_M < cdf_item
                x_L[mask_lo] = x_M[mask_lo]
                x_U[~mask_lo] = x_M[~mask_lo]

                error = np.nanmax(np.abs(F_M - cdf_item))

                if error<tol:
                    break
            
            f_M = p* distrib_end.pdf(x_M) + (1-p) * distrib_end_mix.pdf(x_M)
            error_fin = np.nanmax(np.abs(F_M - cdf_item)/(np.maximum(1e-6, np.abs(f_M))))
            print('Final Error on ppf: ', error_fin)
            #I first get closer by minimizing the CDF difference, then the ppf to make sure that the error in mm is not large

            print('Final error cdf on bisect method: ', error, i)
            nan_mask = np.isnan(F_M)
            x_M[nan_mask] = np.nan
            transf_item_mix = x_M

            transf_item_tot = np.where(np.isnan(transf_item_mix), transf_item, transf_item_mix)
            transf_item = transf_item_tot

        #making sure there are no negative values after the transformation
        neg_values = np.nansum(transf_item.flatten() <0)
        print('Negative values before rect: ', neg_values, neg_values/len(transf_item.flatten()))

        #RECTIFICATION at the end
        if np.isscalar(rect_end)==False:

            thresholds = np.where(np.isinf(rect_end.values), np.nan, rect_end.values)
            lb = 1e-5 * np.ones_like(rect_end.values)
            ub = thresholds
            
            mus = rect_loc_end.values
            stds = rect_std_end.values

            lb_std = (lb - mus)/stds
            ub_std = (ub - mus)/stds
                        
            distrib_rect = truncnorm(lb_std, ub_std, loc = mus, scale = stds)
            
            #Main idea: I replace the data that would be negative (non physical) with the values drawn 
            #from the truncated normal
            rect_data = distrib_rect.ppf(cdf_item/distrib_end.cdf(thresholds))
            transf_item = np.where(transf_item < thresholds, rect_data, transf_item)

            neg_values = np.nansum(transf_item.flatten() <0)
            print('Negative values after rect: ', neg_values, neg_values/len(transf_item.flatten()))

            small_vals_array = np.random.uniform(0, 0.2, size = transf_item.shape)
            transf_item = np.where(transf_item<0, small_vals_array, transf_item)

            neg_values = np.nansum(transf_item.flatten() <0)
            print('Negative values end: ', neg_values, neg_values/len(transf_item.flatten()))
        # archiving
        out.append((transf_item, scen))

    return out


def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights

    Source:
      https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
      @author Jack Peterson (jack@tinybike.net)
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)

    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx : idx + 2])
        else:
            w_median = s_data[idx + 1]
    return w_median


def listxrds_to_np(listds, name_var, forcescen, coords=None):
    """
    **Temporary** function to prepare the format of inputs for training. This is meant
    to be used ONLY before the format of MESMERv1 data is confirmed.

    Parameters
    ----------
    listds : list of xr Dataset
        predictors or target.
    name_var : str
        name of the variable to select.
    forcescen : list of str
        names of the scenarios to select, in this specific order. used to ensure the
        consistency between predictors & target.
    coords : dict
        default None (no selection). Otherwise, will loop over keys & values of
        dictionary to select in listds.
    """
    # looping over scenarios
    tmp = []

    for scen in forcescen:

        # could be replaced with a while, but still a quick loop
        for item in listds:

            if item[1] == scen:
                # for each scenario, creating one unique series: consistency of members &
                # scenarios have to be ensured while loading data
                if coords is not None:
                    tmp.append(item[0][name_var].loc[coords].values.flatten())
                else:
                    tmp.append(item[0][name_var].values.flatten())

    return np.hstack(tmp)
