#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from PharmaPy import xp as np
import json
import re

from PharmaPy.Commons import get_permutation_indexes
from PharmaPy.Errors import PharmaPyTypeError, PharmaPyValueError

# from autograd import numpy as np

gas_ct = 8.314  # J/mol/K
eps = np.finfo(float).eps


def cryst_mechanism(sup_sat, moms, temp, temp_ref, params, reformulate, kv,
                    order):
    sec = False
    if len(params)==4:
        phi_1, phi_2, exp, s_2 = params
        sec = True
    else:
        phi_1, phi_2, exp = params[:3] #for size dependent growth, len(params) is 5 but the last 2 are used in the fvm_method code
        

    # absup = np.maximum(eps, sup_sat)
    absup_ = np.abs(sup_sat)

    absup = np.maximum(eps, absup_)
    if reformulate:
        pre_exp = np.exp(phi_1 + np.exp(phi_2)*(1/temp_ref - 1/temp))
    else:
        pre_exp = phi_1 * np.exp(-phi_2/gas_ct/temp)

    kinetic_term = pre_exp * sup_sat * absup**(exp - 1)

    if sec:
        if moms.ndim == 1:
            mom = np.maximum(0, moms[order]) # For vector moms

        elif moms.ndim == 2:
            mom= np.maximum(0, moms[:, order]) # for matrix

        if np.is_traced(mom):
            kinetic_term = kinetic_term * _power_from_zero(mom * kv, s_2)
        else:
            kinetic_term *= (mom * kv)**s_2

    return kinetic_term


def _power_from_zero(base, exponent):
    """base**exponent for base >= 0, with a finite derivative at zero.

    The same value as the power. Its derivative at zero is infinite for an
    exponent below one -- secondary nucleation's magma term is
    (kv*mu_3)**s_2 with s_2 = 1e-5, and every batch starts with mu_3 = 0 --
    and an exact Jacobian holding inf stops the integrator. Taken from the
    zero side here, the derivative there is 0.
    """
    positive = base > 0

    return np.where(positive, np.where(positive, base, 1.0) ** exponent,
                    0.0 ** exponent)

def disect_rxns(rxns, sep='-->'):

    out = {}
    species = []

    for ind, rxn in enumerate(rxns):
        out[ind] = {}
        left, right = rxn.split(sep)

        reactants = [x.strip() for x in left.split('+')]
        products = [x.strip() for x in right.split('+')]

        out[ind]['reactants'] = reactants
        out[ind]['products'] = products

        species += reactants
        species += products

    regex = r'^\d+(\.\d+)?(/\d+)?\s?'
    for ind, sp in enumerate(species):
        species[ind] = re.sub(regex, '', sp)

    species = list(dict.fromkeys(species))

    return out, species


def get_coeff(pattern, expr):
    text = re.match(pattern, expr)

    if text is None:
        coeff = 1
    elif '/' in text.group():
        num, denom = text.group().split('/')
        coeff = int(num) / int(denom)
    else:
        coeff = float(text.group())

    return coeff


def get_stoich(di_rxn, partic_species):

    num_rxns = len(di_rxn)
    num_species = len(partic_species)

    # TODO: read json keys (name species) and make

    stoich = np.zeros((num_rxns, num_species))

    # I think this is the right regex pattern...
    regex_coeff = r'\d+(\.\d+)?(/\d+)?'
    regex_sub = r'^\d+(\.d+)?(/\d+)?\s?'

    for num, di in di_rxn.items():
        for r in di['reactants']:
            coeff = get_coeff(regex_coeff, r)

            r = re.sub(regex_sub, '', r)

            col = partic_species.index(r)

            stoich[num, col] = -coeff

        for p in di['products']:
            coeff = get_coeff(regex_coeff, p)

            p = re.sub(regex_sub, '', p)

            col = partic_species.index(p)

            stoich[num, col] = coeff

    return stoich


class RxnKinetics:
    r"""
    Create a reaction kinetics object. Reaction rate r\ :sub:`i` is assumed to
    have the following functional form: 
        r\ :sub:`i` = f\ :sub:`1` (T) * f\ :sub:`2` ( C\ :sub:`1`, ..., C\ :sub:`n_comp`) 
        
    with the temperature-dependent term f\ :sub:`1` given by:
        f\ :sub:`1` = k\ :sub:`i` * exp(- Ea\ :sub:`i`/R/T)

    Composition-dependent term f\ :sub:`2` can be passed as a user-defined
    function. If not given, f\ :sub:`2` is assumed to be of the form:
        f\ :sub:`2` = prod\ :sub:`j in reactants for rxn i` C\ :sub:`j` (alpha\ :sub:`{i,j}`)

    where alpha\ :sub:`{i,j}` values are determined automatically by PharmaPy from
    the stoichiometric matrix of the reaction system. Custom reaction
    orders can also be passed through the 'params_f' argument

    Parameters
    ----------
    path : str
        path to the pure-component json file database
    k_params : list or tuple
        pre-exponential factor value(s) for the temperature-dependent term f\ :sub:`1`.
    ea_params : list or tuple
        activation energy [J/mol] value(s) for the temperature-dependent
        term f\ :sub:`1`.
    rxn_list: list of str, optional.
        list containing reactions represented by strings, where the
        pattern '+' separates reactants or products from one another, and
        the pattern --> separates groups of reactants from groups of
        products. Examples of reactions are

            'A + B --> C'
            '2A --> B'
            '2 H\ :sub:`2` O --> 2 H\ :sub:`2` + O\ :sub:`2`',
            'H\ :sub:`2` O --> H\ :sub:`2` + 0.5 O\ :sub:`2`,
            'H\ :sub:`2` O --> H\ :sub:`2` + 0.5 O\ :sub:`2`'
         

        Note that integer, float and fractional stoichiometric coefficients
        are supported.

        The names used for the reactions have to match those on the
        pure-component json file. If 'rxn_list' is None, then both
        stoichiometric_matrix' and 'partic_species' have to be passed
        (see below). The default is None.
    stoiciometric_matrix : numpy array, optional
        stoichiometric matrix for the set of reactions. It must have
        n_rxn rows and n_comp columns, so the element (i, j) represents
        the coefficient of species j in reaction i
    partic_species : list (or tuple) of str, optional
        names of participating species. It will be assumed that the
        order of the names in 'partic_species' is that of the columns of
        'stoichiometric_matrix'. The passed names must match those
        in the pure-component json file
    keq_params : TYPE, optional
        DESCRIPTION. The default is None.
    params_f : numpy array, optional
        parameters for the concentration-dependent term f\ :sub:`2`.
        If no custom model is provided through the 'kinetic_model'
        argument, then 'params_f' values are interpreted as the reaction
        orders of the built-in elementary reaction kinetic model.
        The params_f argument is optional only if no custom model is provided.
        If not given, the reaction orders are set to the stoichiometric
        coefficients for the involved reactants. The default is None.
    temp_ref : float, optional
        reference temperature [K]. If not passed, it will be set to np.inf.
        The default is None.
    reformulate_kin : bool, optional
        if True, f\ :sub:`1` (T) will be reformulated as:

            f\ :sub:`1` (T) = exp[phi\ :sub:`1` + exp(phi\ :sub:`2`) * (1/T_ref - 1/T)]

        where phi\ :sub:`1` = ln(ki\ :sub:`i`) - Ea/R/T_ref and phi\ :sub:`2` = ln(Ea/R)
        We recommend to use this reparametrization when performing
        parameter estimation with datasets at different temperatures.
        The default is False.
    delta_hrxn : float, optional
        DESCRIPTION. The default is 0.
    tref_hrxn : float, optional
        DESCRIPTION. The default is 298.15.

    kinetic_model : callable, optional  
        kinetic model to be used to compute f\ :sub:`2`. It must have
        the signature:

            >>> kin_model(conc, params, *args). The default is None.

    df_dstates : TYPE, optional
        DESCRIPTION. The default is None.
    df_dtheta : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    RxnKinetics object.

    """
    def __init__(self, path, k_params, ea_params, rxn_list=None,
                 stoich_matrix=None, partic_species=None,
                 temp_ref=None, reformulate_kin=False,
                 keq_params=None, params_f=None, delta_hrxn=0,
                 tref_hrxn=298.15, kinetic_model=None, df_dstates=None,
                 df_dtheta=None):


        with open(path) as f:
            db = json.load(f)

        name_species = list(db.keys())

        # Stoichiometry
        if rxn_list is not None:
            di, partic_species = disect_rxns(rxn_list)
            stoich_matrix = get_stoich(di, partic_species)
        else:
            stoich_matrix = np.atleast_2d(stoich_matrix)
            if partic_species is None:
                raise PharmaPyTypeError('Please provide a participating species list when using a stoichiometric matrix.')

        perm_idx = get_permutation_indexes(name_species, partic_species)
        stoich_matrix = stoich_matrix[:, perm_idx]

        partic_species = [partic_species[ind] for ind in perm_idx]
        self.partic_species = partic_species

        self.num_rxns, self.num_species = stoich_matrix.shape

        if temp_ref is None:
            temp_ref = np.inf

        self.temp_ref = temp_ref
        self.reformulate_kin = reformulate_kin

        self.args_kin = ()

        # ---------- Kinetic model
        self.elem_flag = False
        if kinetic_model is None:
            self.kinetic_model = self.elem_f_model
            self.df_dstates = self.elem_df_dstates
            self.df_dthetaf = self.elem_df_dtheta

            self.elem_flag = True
        else:
            self.kinetic_model = kinetic_model
            self.df_dstates = df_dstates
            self.df_dthetaf = df_dtheta

        # Normalize stoichiometric coefficients
        first_negative = (stoich_matrix < 0).argmax(axis=1)
        ref_stoich = np.zeros(self.num_rxns)

        for ind in range(self.num_rxns):
            ref_stoich[ind] = stoich_matrix[ind, first_negative[ind]]

        self.normalized_stoich = stoich_matrix.T / abs(ref_stoich)
        self.stoich_matrix = stoich_matrix

        # ---------- Parameters
        params_dict = {'k_params': k_params, 'ea_params': ea_params,
                       'keq_params': keq_params, 'params_f': params_f}

        self.set_params(params_dict)
        self.nomenclature(stoich_matrix, k_params)

        # Equilibrium kinetics
        if keq_params is None:
            self.keq_params = keq_params
        else:
            self.keq_params = np.atleast_1d(keq_params)

        # Heat of reaction
        if delta_hrxn is None:
            self.delta_hrxn = delta_hrxn
            self.tref_hrxn = tref_hrxn
        else:
            self.delta_hrxn = np.atleast_1d(delta_hrxn)
            if tref_hrxn is None:
                self.tref_hrxn = temp_ref
            else:
                self.tref_hrxn = tref_hrxn

        # Outputs
        self.rxn_rates = None
        self.time_profile = None
        self.conc_profile = None
        self.sensitivities = None

    def transform_params(self, kvals, evals):
        if self.reformulate_kin:
            ea_term = evals/gas_ct/self.temp_ref

            phi_1 = np.log(kvals) - ea_term
            phi_2 = np.log(evals/gas_ct)

        else:
            phi_1 = kvals
            phi_2 = evals

        return phi_1, phi_2

    def set_params(self, params):  # From 1D params to matrix-shaped params

        if isinstance(params, dict):
            k_params = np.atleast_1d(params['k_params']) + eps
            ea_params = np.atleast_1d(params['ea_params']) + eps

            self.phi_1, self.phi_2 = self.transform_params(k_params, ea_params)

            self.num_paramsk = len(self.phi_1) + len(self.phi_2)

            self.fit_paramsf = True
            if self.elem_flag:
                params_f = params.get('params_f', None)
                if params_f is None:
                    is_reactant = self.stoich_matrix < 0
                    orders = abs(is_reactant * self.stoich_matrix)
                    self.fit_paramsf = False
                else:
                    order_map = self.stoich_matrix < 0

                    params_f = params['params_f']
                    if not isinstance(params_f[0], (list, tuple)):
                        params_f = [params_f]

                    orders = np.zeros_like(self.stoich_matrix)
                    for ind, order in enumerate(params_f):
                        orders[ind, order_map[ind]] = order

                if orders.ndim == 1:
                    orders = orders[np.newaxis, ...]

                params_f = orders
            elif params_f is None:
                raise RuntimeError("For user-defined kinetic function, "
                                   "argument 'params_f' is mandatory.")

            self.params_f = orders
            self.order_map = self.stoich_matrix < 0

        else:
            self.phi_1, self.phi_2 = np.split(params[:self.num_paramsk], 2)

            if self.elem_flag:
                # Reorganize rxn orders into a stoich_matrix-like structure
                if self.fit_paramsf:
                    self.params_f = np.zeros_like(self.stoich_matrix,
                                                  dtype=np.float64)

                    self.params_f[self.order_map] = params[self.num_paramsk:]

    def nomenclature(self, stoich_matrix, kvals):

        # Names
        num_kpar = len(self.phi_1)

        if self.reformulate_kin:
            name_k = ['\\varphi_{1, %i}' % ind for ind in range(1, num_kpar + 1)]
            name_e = ['\\varphi_{2, %i}' % ind for ind in range(1, num_kpar + 1)]
        else:
            name_k = ['k_%i' % ind for ind in range(1, num_kpar + 1)]
            name_e = ['E_{a, %i}' % ind for ind in range(1, num_kpar + 1)]

        if self.fit_paramsf:
            num_orders = (stoich_matrix < 0).sum()
            name_orders = [r'\alpha_{}'.format(ind)
                           for ind in range(1, num_orders + 1)]
        else:
            name_orders = []

        self.name_params = name_k + name_e + name_orders
        self.num_params = len(self.name_params)
        self.params = dict(zip(self.name_params, (self.phi_1, self.phi_2)))

    # def set_stoichiometry(self, stoich_matrix):

    #     stoich_matrix = np.atleast_2d(stoich_matrix)
    #     self.num_rxns, self.num_species = stoich_matrix.shape

    #     # Normalize stoichiometric coefficients
    #     first_negative = (stoich_matrix < 0).argmax(axis=1)
    #     ref_stoich = np.zeros(self.num_rxns)

    #     for ind in range(self.num_rxns):
    #         ref_stoich[ind] = stoich_matrix[ind, first_negative[ind]]

    #     self.normalized_stoich = stoich_matrix.T / abs(ref_stoich)
    #     self.stoich_matrix = stoich_matrix

    def concat_params(self):

        params_k_conc = np.concatenate((self.phi_1, self.phi_2))
        if self.elem_flag:
            if self.fit_paramsf:  # rxn orders not fixed
                orders = self.params_f[self.order_map]
                params_concat = np.concatenate((params_k_conc, orders))

            else:  # rxn orders are fixed
                params_concat = params_k_conc

        else:
            params_concat = np.concatenate((params_k_conc, self.params_f))

        return params_concat

    def temp_term(self, temp):

        temp = np.asarray(temp)
        inv_temp = (1/self.temp_ref - 1/temp)

        if self.reformulate_kin:

            if temp.ndim == 0:
                k_temp = np.exp(self.phi_1 + np.exp(self.phi_2) * inv_temp)
            else:
                k_temp = np.exp(self.phi_1 +
                                np.outer(inv_temp, np.exp(self.phi_2)))

        else:
            if temp.ndim == 0:
                k_temp = self.phi_1 * np.exp(self.phi_2/gas_ct * inv_temp)
            else:
                k_temp = self.phi_1 * \
                    np.exp(np.outer(inv_temp, self.phi_2/gas_ct))

        return k_temp

    def equil_term(self, temp, deltah_temp):
        temp = np.asarray(temp)
        inv_temp = (1/temp - 1/self.tref_hrxn)

        if temp.ndim == 0:
            k_eq = self.keq_params * np.exp(-deltah_temp/gas_ct * inv_temp)
        else:
            k_eq = self.keq_params * \
                np.exp(-np.outer(inv_temp, deltah_temp/gas_ct))

        return k_eq

    def dk_dkparams(self, temp):
        temp_term = self.temp_term(temp)

        inv_temp = (1/self.temp_ref - 1/temp)

        if self.reformulate_kin:
            drate_dphi1 = np.diag(temp_term)
            dphi_2 = temp_term * inv_temp * np.exp(self.phi_2)
        else:
            drate_dphi1 = np.diag(np.exp(self.phi_2/gas_ct * inv_temp))
            dphi_2 = temp_term/gas_ct * inv_temp

        drate_dphi2 = np.diag(dphi_2)
        drate_dk = np.hstack((drate_dphi1, drate_dphi2))

        return np.atleast_2d(drate_dk)

    def elem_f_model(self, conc, rxn_orders):
        """ Compute elementary reaction rates for each participating reaction

        Parameters
        ----------
        conc : array-like
            molar concentrations for each participating species (size n_comp)

        Returns
        -------
        rxn_rates : array
            rate for each reaction taking place in the system (size n_rxns)
        rates_species : array
            rate for each species in each reaction (size n_rxns x n_comp)
        total_rates : array
            net reaction rate for each component among all the reactions it
            participates in (size n_comp)
        """

        conc = np.maximum(eps, conc)
        f_term = np.exp(np.dot(np.log(conc), rxn_orders.T))
        return f_term

    def equilibrium_model(self, conc, temp, deltah_rxn):
        is_product = self.stoich_matrix > 0
        orders = abs(is_product * self.stoich_matrix)
        conc = np.asarray(conc)
        n_conc = len(conc)

        keq_temp = self.equil_term(temp, deltah_rxn)

        keq_temp[keq_temp == 0] = 1e20

        # Forward term
        f_term = self.elem_f_model(conc, self.params_f)

        # Backward term
        if conc.ndim == 1:
            r_term = np.zeros(self.num_rxns)
            for ind in range(self.num_rxns):
                r_term[ind] = np.prod(conc**(orders[ind])) / keq_temp[ind]
        else:
            r_term = np.zeros((n_conc, self.num_rxns))
            for ind in range(self.num_rxns):
                r_term[:, ind] = np.prod(
                    conc**(orders[ind]), axis=1) / keq_temp[ind]

        overall_rate = f_term - r_term

        return overall_rate

    def elem_df_dstates(self, conc):

        f_term = self.elem_f_model(conc, self.params_f)

        df_dconc = f_term[..., np.newaxis] * self.params_f / (conc + eps)

        return df_dconc

    def elem_df_dtheta(self, conc):

        f_term = self.elem_f_model(conc, self.params_f)

        conc_correc = np.maximum(np.ones_like(conc) * eps, conc)

        num_orders = self.order_map.sum()
        drate_dorder = np.zeros((self.num_rxns, num_orders))

        count = 0

        for ind, row in enumerate(self.order_map):
            conc_m = conc_correc[row]  # see Section 3.2.2

            norder_i = sum(row)
            drate_dorder[ind,
                         count:count + norder_i] = np.log(conc_m) * f_term[ind]

            count += norder_i
        drate_dorder = drate_dorder

        return drate_dorder

    def derivatives(self, conc, temp, dstates=True):
        temp_terms = self.temp_term(temp)
        f_terms = self.kinetic_model(conc, self.params_f, *self.args_kin)

        if dstates:  # --------------- wrt states
            df_dstates = self.df_dstates(conc, *self.args_kin)
            dr_dstates = (temp_terms * df_dstates.T).T
            jac_states = np.dot(self.normalized_stoich, dr_dstates)
            return jac_states
        else:  # --------------- wrt parameters
            dk_dphi = self.dk_dkparams(temp).T
            dr_dthetak = (dk_dphi * f_terms).T

            if self.fit_paramsf:
                dr_dthetaf = self.df_dthetaf(conc, *self.args_kin)
                dr_dthetaf = (dr_dthetaf.T * temp_terms).T

                dr_dparams = np.hstack((dr_dthetak, dr_dthetaf))

            else:
                dr_dparams = dr_dthetak

            jac_params = np.dot(self.normalized_stoich, dr_dparams)

            if jac_params.ndim == 1:
                jac_params = jac_params[..., np.newaxis]

            return jac_params

    def get_rxn_rates(self, conc, temp=298.15, overall_rates=True, jac=False,
                      delta_hrxn=None, max_iter=3, return_both=False,
                      limiter_dt=1.0):

        if jac:
            jac_states = self.derivatives(conc, temp)
            return jac_states

        else:
            temp_terms = self.temp_term(temp)

            if self.keq_params is None:
                f_terms = self.kinetic_model(conc, self.params_f,
                                             *self.args_kin)
            else:
                f_terms = self.equilibrium_model(conc, temp, delta_hrxn)

            rxn_rates = temp_terms * f_terms

            rrates = rxn_rates.copy()

            # Keep a species from being driven negative, by scaling the
            # reaction extents rather than the species rates.
            #
            # One factor applied to EVERY extent, not a per-species fix to the
            # reactions that consume each offender. Scaling all extents by a
            # single number is trivially stoichiometry preserving, leaves the
            # selectivity between reactions exactly as the kinetics set it,
            # and terminates without iterating -- so there is no max_iter to
            # exhaust and no way to return a rate set that still goes
            # negative. It is the more conservative choice, because a reaction
            # that does not touch the limiting species is throttled too.
            #
            # limiter_dt is the horizon over which depletion is anticipated,
            # and it is explicit because conc and total_rates are in different
            # units: adding them directly, as this once did, silently assumed
            # a one second window. MultiPhaseVessel passes its own
            # positivity_horizon so the vessel limiter and this one agree.
            #
            # Still irreducibly 1-D: conc[short] selects rows on a 2-D input.
            # Pre-refactor this lived inside `if overall_rates:`, which is why
            # the 2-D post-processing calls (Reactors.py:686, 1037, 1555,
            # 1722, 1732, all overall_rates=False) never reached it. That
            # guard is kept, extended to return_both -- the refactored stack's
            # only caller.
            if (overall_rates or return_both) and np.asarray(conc).ndim == 1:

                total_rates = np.dot(rrates, self.normalized_stoich.T)
                short = conc + total_rates * limiter_dt < -eps * 10

                # The ratio for every species, +inf where it is not short,
                # so the shape does not depend on the values and a trace can
                # follow it. The scale is the one the short species alone
                # give, and exactly 1.0 when nothing is short.
                consumed = -total_rates * limiter_dt
                ratio = np.where(short, conc / np.where(short, consumed, 1.0), np.inf)
                rrates = rrates * np.clip(np.min(ratio), 0.0, 1.0)

            total_rates = np.dot(rrates, self.normalized_stoich.T)
            if return_both:
                return rrates,total_rates
            if overall_rates:  # per species
                return total_rates
            else:  # per rxn
                # Deliberately the raw extents: this path never ran the
                # limiter pre-refactor and the legacy Reactors.py callers
                # depend on getting kinetics untouched.
                return rxn_rates


class CrystKinetics:
    """ Specify a kinetics crystallization kinetics object

        Parameters
        ----------
        coeff_solub : array-like
            coefficients for a temperature-dependent solubility (S) polynomial
            of the form S = A + B*T + C*T^2...
        nucl_prim : array-like (3 elements)
            primary nucleation coefficients, with the result given in
            number of particles per second per cubic meter slurry
        nucl_sec : array-like (4 elements) (optional)
            secondary nucleation coefficients, with the result given in
            number of particles per second per cubic meter slurry
        growth : array-like (dimension 3) (optional)
            nucleation parameters, with the result given in um/s
        dissolution : array_like (dimension 3) (optional)
            dissolution parameters, with the result given in um/s
        rel_super : bool (optional, default True)
            if True, relative supersaturation is used for computing the kinetic
            mechanism
        alpha_fn : callable (optional)
            function that receives the vector of liquid phase compositions
            and calculates the growth inhibition term, returning a float
            between 0 and 1

        Returns
        -------

    """

    # Composition bases a solubility correlation can be expressed in. The
    # correlation itself carries no units, so which one is meant has to be
    # stated -- "mass_conc" is ambiguous because it does not say per what.
    # Whichever is named here is the basis the liquid composition is converted
    # into before being compared against the solubility.
    SOLUBILITY_BASES = (
        'mass_per_volume_solution',   # kg solute / m3 solution
        'mass_per_volume_solvent',    # kg solute / m3 solvent
        'mass_per_mass_solvent',      # kg solute / kg solvent
        'mass_per_mass_solution',     # kg solute / kg solution (= mass_frac)
        'mole_per_volume_solution',   # kmol solute / m3 solution (= mol/L)
        'mole_frac',                  # mole fraction
    )

    def __init__(self, coeff_solub=None, solub_fn=None,
                 nucl_prim=None, nucl_sec=None, growth=None, dissolution=None,
                 solubility_type='polynomial', sup_sat_type='relative',
                 reformulate_kin=False, alpha_fn=None,
                 temp_ref=298.15, custom_mechanisms=None,
                 mu_sec_nucl='volume',
                 solubility_basis='mass_per_volume_solution'):
        """
        Parameters
        ----------
        solub_fn : callable, optional
            function with the signature solub_fn(temp, conc)
        sup_sat_type : string, optional
            Default : 'relative'
            if 'relative', supersaturation is calculated as:
                S = (c - c_sat)/ c_sat.
            if, 'ratio':
                S = c/ c_sat
            if, 'absolute':
                S = c- c_sat.
            where c is instantaneous concentration and c_sat is saturated concentration [kg/m3]
        custom_mechanisms: dict of callables
        mu_sec_nucl : string
            if 'area', mu_2 will be used on the size-dependent term of Bs, else
            if 'volume', mu_3 will be used, for secondary nucleation written as:

                Bs = k_s * S^(s_1) * (mu_sec_nucl * k_v)^(s_2)

        """

        self.target_idx = None

        self.temp_ref = temp_ref
        self.sup_sat_type = sup_sat_type
        self.reformulate_kin = reformulate_kin

        if solubility_basis not in self.SOLUBILITY_BASES:
            raise PharmaPyValueError(
                f"solubility_basis={solubility_basis!r} is not one of "
                f"{list(self.SOLUBILITY_BASES)}"
            )

        # Default matches original PharmaPy, which feeds solubility_temp a
        # mass concentration on a solution-volume basis (Crystallizers.py
        # material_balances, basis='mass_conc').
        self.solubility_basis = solubility_basis

        if solub_fn is None:
            self.get_solubility = self.solubility_temp
        else:
            self.get_solubility = solub_fn

        mu_sec = {'area': 2, 'volume': 3}
        self.mu_sec_nucl = mu_sec[mu_sec_nucl]

        if custom_mechanisms is None:
            custom_mechanisms = {}
        elif not isinstance(custom_mechanisms, dict):
            raise PharmaPyTypeError('Provide a dictionary of callables for kinetics.')

        self.custom_mechanisms = custom_mechanisms

        # ---------- Parameters
        self.names_mechanisms = ('nucl_prim', 'nucl_sec', 'growth',
                                 'dissolution')

        params_all = [nucl_prim, nucl_sec, growth, dissolution]

        params_keys = [name for param, name
                       in zip(params_all, self.names_mechanisms)
                       if param is not None]

        params_list = [item for item in params_all if item is not None]

        self.num_par_tuple = [len(item) for item in params_list]

        param_dict = {key: value
                      for key, value in zip(params_keys, params_list)}

        self.set_params(param_dict)  # create self.params

        self.coeff_solub = np.asarray(coeff_solub)
        self.solub_type = solubility_type

        if reformulate_kin:
            if len(param_dict['growth']) == 3:
                self.name_params = (r'\log(k_{bp})', r'\log(E_{bp}/R)', 'b',
                                    r'\log(k_{bs})', r'\log(E_{bs}/R)', 's_1', 's_2',
                                    r'\log(k_{g})', r'\log(E_{g}/R)', 'g',
                                    r'\log(k_{d})', r'\log(E_{d}/R)', 'd')
            else:
                self.name_params = (r'\log(k_{bp})', r'\log(E_{bp}/R)', 'b',
                                r'\log(k_{bs})', r'\log(E_{bs}/R)', 's_1', 's_2',
                                r'\log(k_{g})', r'\log(E_{g}/R)', 'g','alpha','beta'
                                r'\log(k_{d})', r'\log(E_{d}/R)', 'd')
        else:
            if len(param_dict['growth']) == 3:
                self.name_params = ('k_{bp}', 'E_{bp}', 'b',
                                    'k_{bs}', 'E_{bs}', 's_1', 's_2',
                                    'k_{g}', 'E_{g}', 'g',
                                    'k_{d}', 'E_{d}', 'd')
            else:
                self.name_params = ('k_{bp}', 'E_{bp}', 'b',
                                'k_{bs}', 'E_{bs}', 's_1', 's_2',
                                'k_{g}', 'E_{g}', 'g','alpha','beta',
                                'k_{d}', 'E_{d}', 'd')

        self.num_params = len(self.name_params)

        # ---------- Growth decreasing fn
        if alpha_fn is None:
            self.alpha_fn = lambda conc: 1
        else:
            self.alpha_fn = alpha_fn

    def set_params(self, params_in):
        if isinstance(params_in, dict):
            params_complete = self.transform_params(params_in,
                                                    self.reformulate_kin)
        else:
            param_lens = np.array([3, 4, 3, 3])
            split_idx = param_lens.cumsum()[:-1]

            params_in = np.split(params_in, split_idx)

            params_complete = dict(zip(self.names_mechanisms, params_in))

        self.params = params_complete

    def transform_params(self, param_dict, reparam):
        self.params_sec = param_dict.get('nucl_sec')
        if reparam:
            zero_log = np.log(eps)

            nucl_prim = [zero_log, 0, 0]
            nucl_sec = [zero_log, 0, 0, 0]
            growth = [zero_log, 0, 0]
            dissol = [zero_log, 0, 0]

            params_parsed = {'nucl_prim': nucl_prim, 'nucl_sec': nucl_sec,
                             'growth': growth, 'dissolution': dissol}

            for name in self.names_mechanisms:
                if name in param_dict.keys():
                    vals = param_dict[name]

                    tref = self.temp_ref
                    phi_1 = np.log(vals[0] + eps) - vals[1]/gas_ct/tref
                    phi_2 = np.log((vals[1] + eps)/gas_ct)

                    params_parsed[name] = list(vals)

                    params_parsed[name][0] = phi_1
                    params_parsed[name][1] = phi_2
        else:
            nucl_prim = [0, 0, 0]
            nucl_sec = [0, 0, 0, 0]
            growth = [0, 0, 0]
            dissol = [0, 0, 0]

            params_parsed = {'nucl_prim': nucl_prim, 'nucl_sec': nucl_sec,
                             'growth': growth, 'dissolution': dissol}

            for name in self.names_mechanisms:
                if name in param_dict.keys():
                    vals = param_dict[name]
                    tref = self.temp_ref
                    phi_1 = vals[0]
                    phi_2 = vals[1]

                    params_parsed[name] = list(vals)

                    params_parsed[name][0] = phi_1
                    params_parsed[name][1] = phi_2

        return params_parsed

    def concat_params(self):
        params = [np.array(vals) for vals in self.params.values()]
        params_conc = np.concatenate(params)
        return params_conc

    def solubility_temp(self, temp, conc=None):
        if self.solub_type == 'polynomial':
            int_coeff = np.arange(len(self.coeff_solub))

            temp = np.asarray(temp)
            if temp.ndim == 0:
                c_satur = (temp**int_coeff * self.coeff_solub).sum()
            else:
                temp = temp[..., np.newaxis]
                c_satur = (temp**int_coeff * self.coeff_solub).sum(axis=1)
                
        elif self.solub_type == 'apelblat':
            a1, a2, a3 = self.coeff_solub
            c_satur = np.exp(a1 + a2/temp + a3*np.log(temp))
        else:
            raise NameError("Bad 'solub_type' name. It must be either "
                            "'polynomial' or 'apelblat")

        return c_satur

    def get_kinetics(self, conc, temp, kv_cry,
                     moments=None, nucl_sec_out=False):

        conc_target = conc.T[self.target_idx]

        # Supersaturation
        conc_sat = self.get_solubility(temp, conc)
        sup_sat = (conc_target - conc_sat)

        if self.sup_sat_type == 'relative':
            sup_sat = sup_sat / conc_sat

        if self.sup_sat_type == 'ratio':
            sup_sat = sup_sat / conc_sat + 1

        par_p, par_s, par_g, par_d = self.params.values()

        if 'nucl_sec' in self.custom_mechanisms:
            args_sec = ()
            par_sec = self.params_sec
        else:
            args_sec = (kv_cry, self.reformulate_kin)
            par_sec = par_s

        if np.is_traced((sup_sat, moments, temp)):
            mechs = self._mechanisms_traced(sup_sat, conc_sat, moments, temp,
                                            kv_cry)

        elif isinstance(sup_sat, float):
            # print(sup_sat)
            args = [sup_sat, conc_sat, moments, temp, self.temp_ref]
            if sup_sat >= 0:

                subset_mech = ('nucl_prim', 'nucl_sec', 'growth')
            else:

                subset_mech = ('dissolution', )

            mechs = {}

            for name in subset_mech:
                if name in self.custom_mechanisms:
                    args_concat = args + [self.params[name]]
                    mechs[name] = self.custom_mechanisms[name](*args_concat)
                else:
                    mechs[name] = cryst_mechanism(sup_sat, moments, temp,
                                                  self.temp_ref,
                                                  self.params[name],
                                                  self.reformulate_kin,
                                                  kv_cry, self.mu_sec_nucl)

            for ky in self.names_mechanisms:
                if ky not in mechs:
                    mechs[ky] = 0

            # Returns
            self.prim_nucl = mechs['nucl_prim']
            self.sec_nucl = mechs['nucl_sec']
            self.growth = mechs['growth']  # um/s
            self.dissol = mechs['dissolution']  # um/s

        else:
            # Divide positive and negative supersaturation periods
            positive_map = sup_sat > 0

            sup_positive = sup_sat[positive_map]
            sup_negative = sup_sat[~positive_map]

            temp_positive = temp[positive_map]
            temp_negative = temp[~positive_map]

            conc_sat_positive = conc_sat[positive_map]
            conc_sat_negative = conc_sat[~positive_map]

            moments_positive = moments[positive_map]
            moments_negative = moments[~positive_map]

            subset_mech = ('nucl_prim', 'nucl_sec', 'growth')

            mechs = {}

            for name in self.names_mechanisms:
                mechs[name] = np.zeros_like(sup_sat)

            args = [sup_positive, conc_sat_positive, moments_positive,
                    temp_positive, self.temp_ref]

            for name in subset_mech:
                if name in self.custom_mechanisms:
                    args_concat = args + [self.params[name]]
                    mechs[name][positive_map] = self.custom_mechanisms[name](*args_concat)
                else:
                    mechs[name][positive_map] = cryst_mechanism(sup_positive,
                                                                moments_positive,
                                                                temp_positive,
                                                                self.temp_ref,
                                                                self.params[name],
                                                                self.reformulate_kin,
                                                                kv_cry, self.mu_sec_nucl)

            subset_mech = ('dissolution', )

            args = [sup_negative, conc_sat_negative, moments_negative,
                    temp_negative, self.temp_ref]

            for name in subset_mech:
                if name in self.custom_mechanisms:
                    args_concat = args + [self.params[name]]
                    mechs[name][~positive_map] = self.custom_mechanisms[name](*args_concat)
                else:
                    mechs[name][~positive_map] = cryst_mechanism(sup_negative,
                                                                 moments_negative,
                                                                temp_negative,
                                                                self.temp_ref,
                                                                self.params[name],
                                                                self.reformulate_kin,
                                                                kv_cry, self.mu_sec_nucl)

        if nucl_sec_out:
            return mechs['nucl_prim'], mechs['nucl_sec'], mechs['growth'], mechs['dissolution']
        else:
            nucl = mechs['nucl_prim'] + mechs['nucl_sec']
            return nucl, mechs['growth'], mechs['dissolution']

    def _mechanisms_traced(self, sup_sat, conc_sat, moments, temp, kv_cry):
        """The regime switch in get_kinetics, for a traced supersaturation.

        A trace cannot ask which side of zero the supersaturation is on, so
        both regimes are evaluated and where selects between them, exactly as
        the scalar branch does: nucleation and growth from zero up,
        dissolution below. Each regime sees its own side of zero and a clamp
        at zero on the other, so the branch that is discarded stays finite
        -- custom mechanisms included -- and cannot put a NaN into the
        derivative through where.
        """
        growing = sup_sat >= 0
        sides = {
            True: np.where(growing, sup_sat, 0.0),     # nucleation, growth
            False: np.where(growing, 0.0, sup_sat),    # dissolution
        }

        mechs = {}

        for name in self.names_mechanisms:
            in_growth = name != 'dissolution'
            sup = sides[in_growth]

            if name in self.custom_mechanisms:
                value = self.custom_mechanisms[name](
                    sup, conc_sat, moments, temp, self.temp_ref,
                    self.params[name])
            else:
                value = cryst_mechanism(sup, moments, temp, self.temp_ref,
                                        self.params[name],
                                        self.reformulate_kin, kv_cry,
                                        self.mu_sec_nucl)

            mechs[name] = (np.where(growing, value, 0.0) if in_growth
                           else np.where(growing, 0.0, value))

        self.prim_nucl = mechs['nucl_prim']
        self.sec_nucl = mechs['nucl_sec']
        self.growth = mechs['growth']  # um/s
        self.dissol = mechs['dissolution']  # um/s

        return mechs

    def deriv_cryst(self, conc_tg, conc, temp):
        conc_sat = self.get_solubility(temp, conc)
        ssat = max(eps, (conc_tg - conc_sat) / conc_sat)

        def dmech_dparam(mech, params):
            if self.reformulate_kin:
                phi_2 = params[1]

                dmech = np.array(
                    [mech,
                     mech * (1 / self.temp_ref - 1 / temp) * np.exp(phi_2),
                     mech * np.log(ssat)])
            else:
                e_act = params[1]
                expo = params[2]

                absup = max(eps, ssat)

                dmech = np.array(
                    [np.exp(-e_act/gas_ct/temp) * ssat * absup**(expo - 1),
                     -mech * np.exp(e_act/gas_ct/temp),
                     mech * np.log(ssat)])

            return dmech

        b_par, s_par, g_par, d_par = self.params.values()

        dbp_dpar = dmech_dparam(self.prim_nucl, b_par)
        dbs_dpar = dmech_dparam(self.sec_nucl, s_par)
        dgr_dpar = dmech_dparam(self.growth, g_par)
        ddiss_dpar = dmech_dparam(self.dissol, d_par)

        return dbp_dpar, dbs_dpar, dgr_dpar, ddiss_dpar, conc_sat
    def supports(self, mechanism):
        if not isinstance(mechanism,list):
            mechanism = [mechanism]
        checks = []
        for m in mechanism:
            try:
                checks.append(np.any(self.params[m]))
            except KeyError:
                checks.append(False)
        return any(checks)

