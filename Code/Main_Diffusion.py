# ~~~
# This file is part of the paper:
#
#           " Adaptive Trust Region Reduced Basis Methods for Inverse Parameter Identification Problems "
#
#   https://github.com/michikartmann
#
# Copyright 2023 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Contributors team: Michael Kartmann, Tim Keil
# ~~~
# Description:
# This is the main file for the numerical experiment in section 4: the reconstruction of the diffusion coefficient.

import numpy as np
import matplotlib.pyplot as plt
from pymor.basic import *
from IRGNM import IRGNM_FOM, Qr_IRGNM
from Qr_Vr_TR_IRGNM import Qr_Vr_TR_IRGNM
from problems import whole_problem
from discretizer import discretize_stationary_IP
from pymor.parameters.base import ParameterSpace
from helpers import postprocess

#%% choose exact parameter

####################################
# The parameter for Section 4.3 is 'PacMan', 
# while the one for Section 4.5 is 'other'.
###################################
exact_parameter = 'PacMan' #'other'

#%% Set options    
    
# choose optimization methods
opt_methods = {     
              "Qr IRGNM",
              "FOM IRGNM",
              "Qr-Vr TR IRGNM",
               }

# choose norm in the parameter space
norm_type = 'H1'

# plot and save options
plot_parameters = True
save_plots = True
save_path = 'Diffusion_Plots/'

# general options           
N = 300                                                                        # FE Dofs = (N+1)^2                                                
tol = 1e-14                                                                    # safeguard tolerance
tau = 3.5                                                                      # discrepancy parameter
k_max = 50                                                                     # maxit
noise_level = 1e-5                                                             # noise level  

#%% Create analytical problem and discretize it

# define analytical PDE problem
print('Construct problem..')                                                     
analytical_problem, q_exact, N, problem_type, exact_analytical_problem, energy_problem = whole_problem(
                                                        N = N,
                                                        parameter_location = 'diffusion',
                                                        boundary_conditions = 'dirichlet',
                                                        exact_parameter = exact_parameter,
                                                       )

# define data for the inverse problem and store in opt_data
print('Discretizing problem...')                                                
diameter = np.sqrt(2)/N
grid_type = RectGrid
low = 0.001                                                                     # lower bound value
up = 1e20                                                                       # upper bound value 
par_dim = (N+1)**2                                                              # dimension of parameter space
q_low = low*np.ones((par_dim,))                                                 # lower bound
q_up =  up*np.ones((par_dim,))                                                  # upper bound
bounds = [q_low, q_up]
q0 = 3*np.ones((par_dim,))                                                      # initial guess for IRGNM
q_0 =  3*np.ones((par_dim,))                                                    # regularization center for IRGNM
opt_data = {'FE_dim': (N+1)**2,                                                 # FE dofs
            'par_dim': par_dim,                                                     
            'noise_level': noise_level,
            'q_exact': q_exact,
            'q0': q0,
            'q_0': q_0, 
            'low': low,
            'up': up,
            'noise_percentage': None,   
            'problem_type': problem_type,
            'q_in_h1': True,
            'norm_on_q': norm_type,
            'B_assemble': False,                                                # options to preassemble affine components or not
            'bounds': bounds
            } 

# create parameter space
parameter_space = ParameterSpace(analytical_problem.parameters, [low,up]) 

# discretize analytical problem to obtain inverse problem fom
fom_IP, fom_IP_data = discretize_stationary_IP(analytical_problem,
                                               diameter = diameter,
                                               opt_data = opt_data,
                                               exact_analytical_problem = exact_analytical_problem,
                                               energy_product_problem = energy_problem,
                                               grid_type = grid_type
                                               )

#%% Solver setup

cg_tol = 1e-13                                                                  # tolerance for the cg algorithm
cg_maxit = None                                                                 # maxiter for the cg algorithm
theta = .4                                                                      # lower parameter for the acceptance of regularization parameter alpha
Theta = .9                                                                      # upper parameter for the acceptance of regularization parameter alpha
relax_factor = 0                                                                # relax the error estimate
alpha0 = 1e-3                                                                   # initial regularization
reductor_type = 'simple_coercive'                                               # choose pyMOR reductor type

# FOM IRGNM setup
IRGNM_setup = {'display_iteration_process': 1,                                 
              'tau': tau,
              'tol': tol,
              'k_max': k_max,
              'bounds': bounds,
              'cg_tol': cg_tol,
              'cg_maxit': cg_maxit,
              'theta': theta,
              'Theta': Theta,                                                   
              'relax_factor': relax_factor,                                     
              'alpha0': alpha0
              }

# Setup for the inner IRGNM in Qr IRGNM
Inner_IRGNM_setupQr = {'display_iteration_process': 0,                         
                    'tau': 1,
                    'tol': tol,
                    'k_max': 2,
                    'noise_level': noise_level,    
                    'bounds': bounds,
                    'cg_tol': cg_tol,
                    'cg_maxit': cg_maxit,
                    'theta': theta,
                    'Theta': Theta, 
                    'alpha0': alpha0
                     }

IRGNM_setup['Inner_IRGNM_setupQr'] = Inner_IRGNM_setupQr

# Setup for the inner IRGNM in Qr-Vr TR IRGNM
Inner_IRGNM_setupQrVr = {'display_iteration_process': 1,                           
                    'tau': 1,
                    'tol': tol,
                    'k_max': 40,
                    'noise_level': noise_level,    
                    'bounds': bounds,
                    'cg_tol': cg_tol,
                    'cg_maxit': cg_maxit,
                    'theta': theta,
                    'Theta': Theta, 
                    'alpha0': alpha0
                     }

# Setup for Qr-Vr TR IRGNM
TR_IRGNM_setup = {'Inner_IRGNM_setup': Inner_IRGNM_setupQrVr,
                 'tau': tau,                                                    # discrepancy parameter
                 'noise_level': noise_level,                                    # moise level
                 'theta': theta,                                                # lower bound for the choice of the regularization
                 'Theta': Theta,                                                # upper bound for the choice of the regularization
                 'beta': .95,                                                   # trust region boundary parameter
                 'radius': 1,                                                   # init trust region radius
                 'shrink_factor': .5,                                
                 'miniter':0, 
                 'maxiter':k_max, 
                 'tol': tol,                                                    # additional tolerance for usage with nosie level 0
                 'radius_tol':.75,                                              # treshhold for increasing radius
                 'line_search_params': None, 
                 'stagnation_window': 3,
                 'stagnation_threshold': np.inf, 
                 'return_stages': False, 
                 'return_subproblem_stages': False,
                 'reductor_type': reductor_type,
                 'bounds': bounds,
                 'display_iteration_process': 1,
                 'outer_regularization': False,                                 # use outer regularization
                 'alpha_init': alpha0,                                          # initial alpha for first inner IRGNM or outer regularization
                 'relax_factor': relax_factor,
                 'alpha0': alpha0
                 }

q_Qr = q_FOM = q_Qr_Vr = history_QFOM = history_FOM = history_QTRRB = None

#%% Solve the inverse problem

#========== Qr IRGNM ==========================================================s
if 'Qr IRGNM' in opt_methods:
    q_Qr, q_Qr_r, history_QFOM, Qfom_IP, basis_Q = Qr_IRGNM(IRGNM_setup, fom_IP, tol, q0)

#=========== FOM IRGNM ========================================================
if 'FOM IRGNM' in opt_methods:
    q_FOM, history_FOM = IRGNM_FOM(IRGNM_setup, fom_IP, tol, q0)
    
#=============== Qr-Vr TR IRGNM ===============================================
if 'Qr-Vr TR IRGNM' in opt_methods:
    q_Qr_Vr, history_QTRRB = Qr_Vr_TR_IRGNM(parameter_space, fom_IP, TR_IRGNM_setup, q0, tol)  

#%% Visualization

postprocess(tol, tau, noise_level, problem_type, norm_type, N, 
            save_plots, save_path, plot_parameters, 
            fom_IP, q_exact, opt_methods,
            q_FOM, q_Qr, q_Qr_Vr, 
            history_QTRRB, history_QFOM, history_FOM)
