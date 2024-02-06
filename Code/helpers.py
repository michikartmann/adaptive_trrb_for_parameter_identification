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
# Description: this file provides some helper functions and basic routines.

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres, LinearOperator, cg

#%%
def postprocess(tol, tau, noise_level, problem_type, norm_type, N, save_plots, save_path, plot_parameters, fom_IP, q_exact, opt_methods, q_FOM = None, q_Qr = None, q_Qr_Vr = None, history_QTRRB = None, history_QFOM = None, history_FOM = None):
    print('#######################################################################')
    print('################## ALL RESULTS ########################################')
    print(f'problemtype = {problem_type}, N = {N}, regularization_type = {norm_type}')
    print(f'tau = {tau}, noise_level = {noise_level}.')
    print('#######################################################################')
    print('#######################################################################')

    get_data_for_tables = False
    save_eps = False

    # error est usage plot
    if q_Qr_Vr is not None and 1:
         plt.figure()
         plt.plot(history_QTRRB['fom_solves_for_unassembled_residual_error_est'], '.-', label = 'Online computation of the error estimator')
         plt.plot(history_QTRRB['fom_solves_for_assembling_residual_error_est'],'.-', label = 'Assembly of the error estimator')
         plt.plot(history_QTRRB['true_fom_solves_for_error_est'],'.-', label = 'Mixed strategy')
         plt.legend()
         plt.xlabel("k")
         plt.ylabel("FOM solves")
         max_len = len(history_QTRRB['fom_solves_for_unassembled_residual_error_est'])
         plt.xticks(ticks = range(0,max_len,5), labels = range(0,max_len, 5))
         if save_plots:
             if save_path is not None:
                 title = save_path + problem_type+f'_N={N}_errorest'
             else:
                 title = problem_type+f'_N={N}_errorest.png'
             plt.savefig(title+'.png', bbox_inches='tight')
             plt.savefig(title+'.eps',  bbox_inches='tight', pad_inches=0)
         plt.show()
         
    # get min max
    qmin, qmax = fom_IP.get_min_max(q_exact, q_FOM, q_Qr, q_Qr_Vr)
    colormap_bounds = [qmin, qmax]
    
    # plot exact parameter
    if plot_parameters:
       fom_IP.plot_matplotlib(q_exact, 'Exact parameter', save_plots, problem_type+f'_N={N}_exact', save_path, save_eps = save_eps, colormap_bounds = colormap_bounds)
       # fom_IP.plot_matplotlib(fom_IP.opt_data['u_noise'].to_numpy()[0], 'Noisy data', save_plots, problem_type+f'_N={N}_noisydata', save_path, save_eps = save_eps)
       # fom_IP.plot_matplotlib(fom_IP.opt_data['noise'], 'Noise', save_plots, problem_type+f'_N={N}_noise', save_path, save_eps = save_eps)

    # plot reconstructed parameters and print results
    if q_Qr is not None:
        print(f'Qr IRGNM|| time: {history_QFOM["time"]}, FOM solves: {history_QFOM["number_fom_solves"]}, FOM B applications: {history_QFOM["number_fom_B_operator_applications"]}, iterations: {history_QFOM["k"]}, Basis Q: {history_QFOM["len_Q_basis"]}')
        rel_err_QFOM = fom_IP.Q_norm(q_exact-q_Qr)/fom_IP.Q_norm(q_exact)
        print(f'discrepancy residual = {history_QFOM["output_res"][-1]}, norm_grad = {history_QFOM["non_regularized_gradient_norm"][-1]}, rel. error to exact = {rel_err_QFOM}')
        print('-------------------------------------------------------------------')
        if plot_parameters:
            plt.figure()
            history_QFOM['u'] = fom_IP.state(q_Qr).to_numpy()[0]
            fom_IP.plot_matplotlib(q_Qr, r'$q^{Q_r}$', save_plots, problem_type+f'_N={N}_QFOM', save_path,  save_eps = save_eps, colormap_bounds = colormap_bounds)
            # fom_IP.plot_matplotlib(history_QFOM['u'], r'$u^{Q_r}$', save_plots, problem_type+f'_N={N}_stateQFOM', save_path,  save_eps = save_eps)  
            
        if save_plots:
            if save_path is not None:
                title = save_path + problem_type+f'QFOMarray_N={N}.npy'
            else:
                title = problem_type+f'QFOMarray_N={N}.npy'
            np.save(title, history_QFOM)

    if q_FOM is not None:
        print(f'FOM IRGNM|| time: {history_FOM["time"]}, FOM solves: {history_FOM["number_fom_solves"]}, FOM B applications: {history_FOM["number_fom_B_operator_applications"]}, iterations: {history_FOM["k"]}')
        rel_err_FOM = fom_IP.Q_norm(q_exact-q_FOM)/fom_IP.Q_norm(q_exact)
        print(f'discrepancy residual = {history_FOM["output_res"][-1]}, norm_grad = {history_FOM["non_regularized_gradient_norm"][-1]}, rel. error to exact = {rel_err_FOM}')
        print('-------------------------------------------------------------------')
        if plot_parameters:
            plt.figure()
            fom_IP.plot_matplotlib(q_FOM, r'$q^{FOM}$', save_plots, problem_type+f'_N={N}_FOM', save_path,  save_eps = save_eps, colormap_bounds = colormap_bounds)
            history_FOM['u'] = fom_IP.state(q_FOM).to_numpy()[0]
            # fom_IP.plot_matplotlib(history_FOM['u'], r'$u^{FOM}$', save_plots, problem_type+f'_N={N}_stateFOM', save_path,  save_eps = save_eps)
                
        if save_plots:
            if save_path is not None:
                title = save_path + problem_type+f'FOMarray_N={N}.npy'
            else:
                title = problem_type+f'FOMarray_N={N}.npy'
            np.save(title, history_FOM)

    if q_Qr_Vr is not None:
        print(f'Qr-Vr TR IRGNM|| time: {history_QTRRB["time"]}, FOM solves: {history_QTRRB["number_fom_solves"]}, fom solves for error est: {sum(history_QTRRB["true_fom_solves_for_error_est"])}, FOM B applications: {history_QTRRB["number_fom_B_operator_applications"]}, iterations: {history_QTRRB["k"]}, Basis Q = {history_QTRRB["len_Q_basis"]}, Basis V = {history_QTRRB["len_V_basis"]}')
        rel_err_QTRRB = fom_IP.Q_norm(q_exact-q_Qr_Vr)/fom_IP.Q_norm(q_exact)
        print(f'discrepancy residual = {history_QTRRB["output_res"][-1]}, norm_grad = {history_QTRRB["non_regularized_gradient_norm"][-1]}, rel. error to exact = {rel_err_QTRRB}')
        print('-------------------------------------------------------------------')
        if plot_parameters:
            plt.figure()
            fom_IP.plot_matplotlib(q_Qr_Vr, r'$q^{Q_r-V_r}$', save_plots, problem_type+f'_N={N}_QTRRB', save_path,  save_eps = save_eps, colormap_bounds = colormap_bounds)
            history_QTRRB['u'] = fom_IP.state(q_Qr_Vr).to_numpy()[0]
            # fom_IP.plot_matplotlib(history_QTRRB['u'], r'$u^{Q_r-V_r}$', save_plots, problem_type+f'_N={N}_stateQTRRB', save_path,  save_eps = save_eps)
            
        if save_plots:
            if save_path is not None:
                title = save_path + problem_type+f'TRRBarray_N={N}.npy'
            else:
                title = problem_type+f'TRRBarray_N={N}.npy'
            np.save(title, history_QTRRB)
                

    # combined plots and plot differences    
    if "FOM IRGNM" in opt_methods and 'Qr IRGNM' in opt_methods and 'Qr-Vr TR IRGNM' in opt_methods: 
        
        # subplots
        fom_IP.plot_subplot_para(q_exact, q_FOM, q_Qr, q_Qr_Vr, title = None, save = save_plots, save_title = problem_type+f'_N={N}_subplots_q.png', path = save_path)
        
        # L2 error
        print(f'rel. error norm((q_FOM - q_Qr)/q_FOM)= { fom_IP.Q_norm(q_FOM-q_Qr)/fom_IP.Q_norm(q_FOM)}')
        print(f'rel. error norm((q_Qr - q_Qr_Vr)/q_Qr) = { fom_IP.Q_norm(q_Qr-q_Qr_Vr)/fom_IP.Q_norm(q_Qr)}')
        print(f'rel. error norm((q_FOM - q_Qr_Vr)/q_FOM) = { fom_IP.Q_norm(q_FOM-q_Qr_Vr)/fom_IP.Q_norm(q_FOM)}')
        
        q_FOM_pymor = fom_IP.Q_vector_space.from_numpy(q_FOM)
        q_Qr_pymor = fom_IP.Q_vector_space.from_numpy(q_Qr)
        q_Qr_Vr_pymor = fom_IP.Q_vector_space.from_numpy(q_Qr_Vr)
        q_exact_pymor = fom_IP.Q_vector_space.from_numpy(q_exact)
        q_FOM_norm = fom_IP.primal_model.l2_norm(q_FOM_pymor)
        q_exact_norm = fom_IP.primal_model.l2_norm(q_exact_pymor)
        print(f'rel. error L2 norm((q_FOM - q_Qr)/q_FOM)= {(fom_IP.primal_model.l2_norm(q_FOM_pymor-q_Qr_pymor)/q_FOM_norm)[0]}')
        print(f'rel. error L2 norm((q_Qr - q_Qr_Vr)/q_Qr) = { (fom_IP.primal_model.l2_norm(q_Qr_pymor-q_Qr_Vr_pymor)/fom_IP.primal_model.l2_norm(q_Qr_pymor))[0]}')
        print(f'rel. error L2 norm((q_FOM - q_Qr_Vr)/q_FOM) = { (fom_IP.primal_model.l2_norm(q_FOM_pymor-q_Qr_Vr_pymor)/q_FOM_norm)[0]}')
            
        if get_data_for_tables:
            print(f'rel. error L2 norm((q_exact - q_Qr)/q_exact)= {(fom_IP.primal_model.l2_norm(q_exact_pymor-q_Qr_pymor)/q_exact_norm)[0]}')
            print(f'rel. error L2 norm((q_exact - q_Qr_Vr)/q_exact) = { (fom_IP.primal_model.l2_norm(q_exact_pymor-q_Qr_Vr_pymor)/(q_exact_norm))[0]}')
            print(f'rel. error L2 norm((q_exact - q_FOM)/q_exact) = { (fom_IP.primal_model.l2_norm(q_exact_pymor-q_FOM_pymor)/q_exact_norm)[0]}')
            
            # maximum minimum pointwise error
            print(f'max rel. error (q_FOM - q_Qr)/q_FOM)= { max(abs((q_FOM-q_Qr))/abs(q_FOM))}')
            print(f'max rel. error (q_Qr - q_Qr_Vr)/q_Qr) = {max(abs((q_Qr-q_Qr_Vr))/abs(q_Qr))}')
            print(f'max rel. error (q_FOM - q_Qr_Vr)/q_FOM) = { max(abs((q_FOM-q_Qr_Vr))/abs(q_FOM))}')
            print(f'min rel. error (q_FOM - q_Qr)/q_FOM)= { min(abs((q_FOM-q_Qr))/abs(q_FOM))}')
            print(f'min rel. error (q_Qr - q_Qr_Vr)/q_Qr) = {min(abs((q_Qr-q_Qr_Vr))/abs(q_Qr))}')
            print(f'min rel. error (q_FOM - q_Qr_Vr)/q_FOM) = { min(abs((q_FOM-q_Qr_Vr))/abs(q_FOM))}')
        
        # relative differences pointwise
        abs_min, abs_max = fom_IP.get_min_max(abs((q_FOM-q_Qr)), abs((q_Qr-q_Qr_Vr)),abs((q_FOM-q_Qr_Vr)))
        rel_min, rel_max = fom_IP.get_min_max(abs((q_FOM-q_Qr))/abs(q_FOM), abs((q_Qr-q_Qr_Vr))/abs(q_Qr), abs((q_FOM-q_Qr_Vr))/abs(q_FOM))
        rel_min_ex, rel_max_ex = fom_IP.get_min_max(abs((q_exact-q_Qr))/abs(q_exact), abs((q_exact-q_FOM))/abs(q_exact), abs((q_exact-q_Qr_Vr))/abs(q_exact))
        if get_data_for_tables:
            print(f'colorbar abs error to FOM: {abs_min} {abs_max}')
            print(f'colorbar rel error to FOM: {rel_min} {rel_max}')
            print(f'colorbar rel error to exact: {rel_min_ex} {rel_max_ex}')
    
        plt.figure()
        fom_IP.plot_matplotlib(abs((q_FOM-q_Qr))/abs(q_FOM), r'rel error FOM - ${Q_r}$', save_plots, problem_type+f'_N={N}_rel_diff2_fom-qfom' , save_path,  save_eps = save_eps, colormap_bounds = [rel_min, rel_max ])
        plt.figure()
        fom_IP.plot_matplotlib(abs((q_Qr-q_Qr_Vr))/abs(q_Qr), r'rel error ${Q_r}$-${Q_r-V_r}$', save_plots, problem_type+f'_N={N}_rel_diff2_qfom-qtrrb', save_path,  save_eps = save_eps, colormap_bounds = [rel_min, rel_max ] )
        plt.figure()
        fom_IP.plot_matplotlib(abs((q_FOM-q_Qr_Vr))/abs(q_FOM), r'rel error FOM - ${Q_r-V_r}$', save_plots, problem_type+f'_N={N}_rel_diff2_fom-qtrrb', save_path,  save_eps = save_eps, colormap_bounds = [rel_min, rel_max ] )
        
        # # abs differences pointwise
        # plt.figure()
        # fom_IP.plot_matplotlib(abs((q_FOM-q_Qr)), r'abs $d^{Q_r}$', save_plots, problem_type+f'_N={N}_abs_diff2_fom-qfom' , save_path,  save_eps = save_eps, colormap_bounds = [abs_min, abs_max ] )
        # plt.figure()
        # fom_IP.plot_matplotlib(abs((q_Qr-q_Qr_Vr)), r'abs $d^{Q_r,Q_r-V_r}$', save_plots, problem_type+f'_N={N}_abs_diff2_qfom-qtrrb', save_path,  save_eps = save_eps, colormap_bounds = [abs_min, abs_max ]  )
        # plt.figure()
        # fom_IP.plot_matplotlib(abs((q_FOM-q_Qr_Vr)), r'abs $d^{Q_r-V_r}$', save_plots, problem_type+f'_N={N}_abs_diff2_fom-qtrrb', save_path,  save_eps = save_eps, colormap_bounds = [abs_min, abs_max ]  )
        
        # rel error to exact
        plt.figure()
        fom_IP.plot_matplotlib(abs((q_exact-q_Qr))/abs(q_exact), r'rel error exact - ${Q_r}$', save_plots, problem_type+f'_N={N}_rel_diff2_exact-qfom' , save_path,  save_eps = save_eps, colormap_bounds = [rel_min_ex, rel_max_ex ])
        plt.figure()
        fom_IP.plot_matplotlib(abs((q_exact-q_Qr_Vr))/abs(q_exact), r'rel error exact - ${Q_r-V_r}$', save_plots, problem_type+f'_N={N}_rel_diff2_exact-qtrrb', save_path,  save_eps = save_eps, colormap_bounds = [rel_min_ex, rel_max_ex ])
        plt.figure()
        fom_IP.plot_matplotlib(abs((q_exact-q_FOM))/abs(q_exact), r'rel error exact - FOM', save_plots, problem_type+f'_N={N}_rel_diff2_exact-fom', save_path,  save_eps = save_eps, colormap_bounds = [rel_min_ex, rel_max_ex ] )
        
        
    # plot output residual against iterations
    plt.figure()
    if 'Qr IRGNM' in opt_methods:
        plt.semilogy(history_QFOM["output_res"], '-o', label= r'$Q_r$ IRGNM')
    if 'Qr-Vr TR IRGNM' in opt_methods:
        plt.semilogy(history_QTRRB["output_res"], '-o', label= r'$Q_r$-$V_r$ IRGNM')
    if "FOM IRGNM" in opt_methods:
        plt.semilogy(history_FOM["output_res"], '-o', label= 'FOM IRGNM')
    value = tol+noise_level*tau
    plt.axhline(y=value, color='g', linestyle='-', label = r'$\tau\delta$')
    plt.xlabel('$k$')
    plt.ylabel(r'$\|\|F(q^k)-y^{\delta}\|\|_{H}$')
    plt.legend(loc = 1)
    if save_plots:
        if save_path is not None:
            title = save_path + problem_type+f'_N={N}_residual_iteration.png'
        else:
            title = problem_type+f'_N={N}_residual_iteration.png'
        plt.savefig(title, bbox_inches='tight')

    # plot output residual against cpu time
    plt.figure()
    if 'Qr IRGNM' in opt_methods:
        plt.semilogy(history_QFOM["time_steps"], history_QFOM["output_res"], '-o', label= r'$Q_r$ IRGNM')
    if 'Qr-Vr TR IRGNM' in opt_methods:
        plt.semilogy(history_QTRRB["time_steps"], history_QTRRB["output_res"], '-o', label= r'$Q_r$-$V_r$ IRGNM')
    if "FOM IRGNM" in opt_methods:
        plt.semilogy(history_FOM["time_steps"], history_FOM["output_res"], '-o', label= 'FOM IRGNM')
    value = tol+noise_level*tau
    plt.axhline(y=value, color='g', linestyle='-', label = r'$\tau\delta$')
    plt.xlabel('time [s]')
    plt.ylabel(r'$\|\|F(q^k)-y^{\delta}\|\|_{H}$')
    plt.legend(loc = 1)
    if save_plots:
        if save_path is not None:
            title = save_path + problem_type+f'_N={N}_residual_cpu.png'
        else:
            title = problem_type+f'_N={N}_residual_cpu.png'
        plt.savefig(title, bbox_inches='tight')

    if get_data_for_tables:
        print(f'colorbar reconstructions: {fom_IP.get_min_max(q_exact, q_FOM, q_Qr, q_Qr_Vr)}')
        
        for i in range(len(history_QFOM['time_steps'])):
            print(f'{history_QFOM["time_steps"][i]}  {history_QFOM["output_res"][i]}')
        print('------------')
        for i in range(len(history_QTRRB['time_steps'])):
            print(f'{history_QTRRB["time_steps"][i]}  {history_QTRRB["output_res"][i]}')
        print('------------')
        for i in range(len(history_FOM['time_steps'])):
            print(f'{history_FOM["time_steps"][i]}  {history_FOM["output_res"][i]}')
        print('------------')
            
    # # plot gradient norm against iterations
    # plt.figure()
    # if 'Qr IRGNM' in opt_methods:
    #     plt.semilogy(history_QFOM["non_regularized_gradient_norm"], '-o', label= r'$Q_r$ IRGNM')
    # if 'Qr-Vr TR IRGNM' in opt_methods:
    #     plt.semilogy(history_QTRRB["non_regularized_gradient_norm"],'-o', label= r'$Q_r$-$V_r$ IRGNM')
    # if "FOM IRGNM" in opt_methods:
    #     plt.semilogy(history_FOM["regularized_gradi ent_norm"], '-o', label= 'FOM IRGNM nonregularized')
    # plt.xlabel('$k$')
    # plt.legend(loc = 1)
    # plt.title("Gradient norm")
    # if save_plots:
    #     if save_path is not None:
    #         title = save_path + problem_type+f'_N={N}_gradient.png'
    #     else:
    #         title = problem_type+f'_N={N}_gradient.png'
    #     plt.savefig(title, bbox_inches='tight')

#%% projected newton and newton method

def LINSOLVER_with_callback(A, b, tol, solvertype):
      num_iters = 0

      def callback(xk):
         nonlocal num_iters
         num_iters+=1
      if solvertype == 'GMRES':
         x, status= gmres(A, b, x0=None,tol=tol, callback = callback)
      elif solvertype == 'CG':
         x, status= cg(A, b, x0=None, tol=tol, callback = callback)
      else:
          print('ERROR choose linear solver to be CG or GMRES!')
      return x,status,num_iters

def proj_newton(f,df,ddf,x0,opt,low,up,tikonov_stabilization_start,tikonov_stabilization_end):
    
    # read op
    tol = opt["tol"]
    maxit = opt["maxit"]
    tol_lin = opt["tol_lin"]
    eval_norm = opt["norm"]
    inner_tikonov = opt["inner_tikonov"]
    
    # check if bounds are equal
    if np.array_equal(low,up):
       hist = {}
       hist["success"] = 0
       hist["exit_flag"] = 'ProjNewton: Bounds coincided, return bound with res {res:3.3e}.'      
       hist["it"] = 0
       return low , hist
   
    # initialize
    x = x0.copy()
    dim = x0.size 
    k = 0
    if inner_tikonov == True:
        tikonov_stabilization = tikonov_stabilization_start
    else:
        tikonov_stabilization = tikonov_stabilization_end
    rhs = -df(x,tikonov_stabilization)
    norm = eval_norm(rhs)
    norm0 = norm
    res = 1
    hist = {
             "res_rel": [1],
             "linear_solver_iterations": 0
            }
    
    while res >= max(tol*norm0,tol) and k<maxit:
          
          # compute search direction sd
          Active = np.minimum(1, (up-x <= 1e-13)+(x-low <= 1e-13))
          Inactive = np.ones(dim)-Active
          
          if np.sum(Inactive) == 0:
            sd = rhs #take gradient step
          else:
            red_hessian_action = lambda d: Active*d+Inactive*ddf(x,Inactive*d,tikonov_stabilization)
            DDF = LinearOperator( (dim,dim), matvec = red_hessian_action ) 
            sd,exitflag, iter_linear_solver = LINSOLVER_with_callback(DDF,rhs, tol_lin, opt['linear_solver'] )  
            hist["linear_solver_iterations"] += iter_linear_solver
          
          s = 1
          
          # update
          if inner_tikonov == True:
              tikonov_stabilization = max(tikonov_stabilization/4,tikonov_stabilization_end)
             
          x = np.maximum(low, np.minimum(up, x + s * sd)).reshape(dim) 
          
          rhs = -df(x,tikonov_stabilization)
          x_1 = np.maximum(low, np.minimum(up, x + rhs)).reshape(dim) 
          res = eval_norm(x-x_1)
          k += 1
          
          # update hist
          hist["res_rel"].append(res)
          
    if k == maxit:
        flag = f'ProjNewton: Maxit {maxit:<3} reached with res {res:3.3e}.'
        hist["success"] = 0
    else:
        flag = f'ProjNewton: Tol of {res:3.3e} reached after {k:<3} iterations.'   
        hist["success"] = 1
    
    hist["exit_flag"] = flag       
    hist["it"] = k
    return x, hist

def newton(f,df,x0,opt):
    # read opt
    tol = opt["tol"]
    maxit = opt["maxit"]
    tol_lin = opt["tol_lin"]
    eval_norm = opt["norm"]
    
    # initialize
    x = x0
    dim = x0.size 
    k = 0
    rhs_f = -f(x)
    norm = eval_norm(rhs_f)
    norm0 = norm
    hist = {
             "res_rel": [1],
            }
    
    while norm >= max(tol*norm0,tol) and k<maxit:
          
          # compute search direction sd
          DF = LinearOperator( (dim,dim), matvec =lambda d: df(x,d) ) #
          sd,exitflag = gmres(DF,rhs_f,tol = tol_lin)###
          # compute steplength s
          s = 1
          
          # update
          x += s*sd 
          rhs_f = -f(x)
          norm = eval_norm(rhs_f)
          k += 1
          
          # update hist
          hist["res_rel"].append(norm/norm0)
          
    if k == maxit:
        flag = 'Newton Maxit Reached'
        hist["success"] = 0
    else:
        flag = 'Newton Tol Reached'   
        hist["success"] = 1
    
    hist["flag"] = flag       
    hist["it"] = k
    return x, hist


#%% finite difference derivative test
 
def test_derivative(f,df,dim, mode=1):
    
    # Initialize
    Eps = np.array([10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8])
    u  = np.random.random(dim)
    du = np.random.random(dim)
    T = np.zeros(np.shape(Eps))
    T2 = T
    ff = f(u)
    
    # Compute central & right-side difference quotient
    for i in range(len(Eps)):
        #print(Eps[i])
        f_plus = f(u+Eps[i]*du)
        f_minus = f(u-Eps[i]*du)
        if mode == 1:
            T[i] = np.linalg.norm( ( (f_plus - f_minus)/(2*Eps[i]) ) - df(u)*du )
            T2[i] =  np.linalg.norm( ( (f_plus - ff)/(Eps[i]) ) - df(u)*du )
        else:
            T[i] = np.linalg.norm( ( (f_plus - f_minus)/(2*Eps[i]) ) - df(u,du) )
            T2[i] =  np.linalg.norm( ( (f_plus - ff)/(Eps[i]) ) - df(u,du) )
        
    #Plot
    plt.figure()
    plt.xlabel('$eps$')
    plt.ylabel('$J$')
    plt.loglog(Eps, Eps**2, label='O(eps^2)')
    plt.loglog(Eps, T,'ro--', label='Test')
    plt.legend(loc='upper left')
    plt.grid()
    plt.title("Central difference quotient")
    plt.figure()
    plt.xlabel('$eps$')
    plt.ylabel('$J$')
    plt.loglog(Eps, Eps, label='O(eps)')
    plt.loglog(Eps, T2, 'ro--',label='Test')
    plt.legend(loc='upper left')
    plt.grid()
    plt.title("Rightside difference quotient")
    # print(T)
    # print(Eps)
    # print(Eps**2)
    return T


#%% cg

def conjugate_gradient(Afun,f,d0,maxit = None,tol = 1e-13):
    
    dim = len(f)
    if maxit == None:
       maxit = dim
    
    # initialize
    d = d0.copy()
    Ad, ulin0, plin_partial0 = Afun(d)
    if maxit == 0:
        return Ad, {}
    r = f-Ad                                                        
    p = r                                                              
    res_norm = r.T@r                                                     
    i = 0           
    history = {'iterates':[d]}                                                      
    while np.sqrt(res_norm) >= tol and i <= maxit:   
          Ap, ulin, plin_partial = Afun(p)                                    
          alpha = res_norm/(p.T@Ap)                                            
          d += alpha*p                                                   
          r = r-alpha*Ap                                            
          res_norm_new = r.T@ r                                    
          beta = res_norm_new/res_norm                                        
          p = r+ beta*p
          res_norm = res_norm_new                                               
          history['iterates'].append(d)    
          i += 1  
          
    if i == maxit:
        flag = f'CG reached maxiter of i = {i:<3}'\
               f' with res = {np.sqrt(res_norm):1.4e}'  
    else:
        flag = f'CG converged at i = {i:<3}'\
               f' with res = {np.sqrt(res_norm):1.4e}'  
    # print(flag)
    history['i'] = i
    history['res_norm'] = res_norm
    history['exit_flag'] = flag
    return d, history

#%% plot and save iterate

def plot_and_save_iterate(q, string, save = False):
    dim = int(np.sqrt(q.size))
    plt.figure()
    q_cube = q.reshape((dim,dim))
    plt.imshow(q_cube, origin='lower')
    plt.colorbar()
    plt.title(string)
    if save:
        plt.savefig(string+'.png')
    plt.show()  

#%% partition of unity

# Note:
# The code for the partition of unity was taken from Andreas Buhr and can be 
# found at https://doi.org/10.5281/zenodo.2555624 .

def hatfunction(start, peak, end):
    """get a hat function

    1             /\
    |            /  \
    |           /    \
    |          /      \
    0----------        -------------
    |
               |   |   |
           start peak  end

    :param start: value where function starts to rise, may be -float('inf')
    :param peak: value where functions has its maximum
    :param end: value where functions ends to fall, may be float('inf')

    :returns: python function
    """
    start = float(start)
    peak = float(peak)
    end = float(end)
    assert start < peak < end
    assert np.isfinite(peak)

    def f(x):
        x = np.array(x)
        return np.maximum(0,
                   np.minimum(
                       (x - peak) / (peak - start) + 1,
                       (x - peak) / (peak - end) + 1
                   )
                   )
    return f


def ndhatfunction(definitions):
    """get an n dimensional hat function

    :param definitions: list of triples (start,peak,end)

    :returns: python function
    """
    onedfunctions = [hatfunction(*d) for d in definitions]

    def ndf(x):
        x = np.array(x)
        result = np.ones(x.shape[:-1])
        for i, f in enumerate(onedfunctions):
            result *= f(x[..., i])
        return result
    return ndf

def twodhatfunction(domains):
    """get an n dimensional hat function

    :param definitions: list of triples (start,peak,end)

    :returns: python function
    """
    onedfunctions = [hatfunction(*domain) for domain in domains]

    def twodf(x):
        x = np.array(x)
        result = np.ones(x.shape[:-1])
        for i, f in enumerate(onedfunctions):
            result *= f(x[..., i])
        return result
    return twodf

def gen_definitions(boundaries, outer_constant):
    """generates hat function definitions from domain boundaries

    examples:

    (0,1,2,3,4) becomes with outer_constant=False
    [(-float('inf'),0,1),
     (0,1,2),
     (1,2,3),
     (3,4,float('inf'))]

    (0,1,2,3,4) becomes with outer_constant=True
    [(-float('inf'),1,2),
     (2,3,float('inf'))]


    """
    boundaries = list(boundaries)  # create a copy
    if outer_constant:
        boundaries = boundaries[1:-1]

    boundaries = [-float('inf')] + boundaries + [float('inf')]

    if len(boundaries) == 2:
        # we need a constant one function
        return [(-float('inf'), 0, float('inf'))]

    return [(boundaries[i], boundaries[i + 1], boundaries[i + 2])
            for i in range(len(boundaries) - 2)]


def partition_of_unity(boundary_lists, outer_constant=True):
    """get a partition of unity

    :param boundary_lists: a list of lists. For each dimension, a
                           list of domain boundaries
    :param outer_constant: whether the outmost function should be
                           one at the boundary and then go to zero within
                           the first domain or
                           constant one in the first domain and go to zero
                           within the second domain
    :returns: numpy array of python functions
    """

    definition_lists = [gen_definitions(b, outer_constant)
                        for b in boundary_lists]
    resultshape = [len(d) for d in definition_lists]
    alldefs = [[definition] for definition in definition_lists[0]]
    # outer product
    for deflist in definition_lists[1:]:
        alldefs = [oldlist + [newelem]
                   for oldlist in alldefs for newelem in deflist]

    allfuns = map(ndhatfunction, alldefs)
    result = np.array(list(allfuns))
    result = result.reshape(resultshape)
    return result

def localized_pou(coarse_grid_resolution):
    boundaries_x = np.linspace(0., 1., coarse_grid_resolution[0]+1)
    boundaries_y = np.linspace(0., 1., coarse_grid_resolution[1]+1)
    pou = partition_of_unity((boundaries_x, boundaries_y), False)
    return pou

if __name__ == "__main__":
    # most simple test
    myfun = hatfunction(3, 4, 5)
    assert myfun(2) == 0
    assert myfun(3) == 0
    assert myfun(3.5) == 0.5
    assert myfun(4) == 1
    assert myfun(4.5) == 0.5
    assert myfun(5) == 0
    assert myfun(6) == 0
    # vectorized test
    assert np.all(myfun((3.5, 4, 4.5)) == (0.5, 1, 0.5))

    # 'inf' test
    myfun = hatfunction(3, 4, float('inf'))
    assert myfun(2) == 0
    assert myfun(3) == 0
    assert myfun(3.5) == 0.5
    assert myfun(4) == 1
    assert myfun(4.5) == 1
    assert myfun(5) == 1
    assert myfun(6) == 1

    myfun = hatfunction(-float('inf'), 4, 6)
    assert myfun(2) == 1
    assert myfun(3) == 1
    assert myfun(3.5) == 1
    assert myfun(4) == 1
    assert myfun(4.5) == 0.75
    assert myfun(5) == 0.5
    assert myfun(6) == 0

    # n dimensional test
    myfun = ndhatfunction(((3, 4, 5), (4, 5, float('inf'))))
    assert myfun((3, 5)) == 0
    assert myfun((3.5, 5)) == 0.5
    assert myfun((4, 4.5)) == 0.5
    # vectorized test
    assert np.all(myfun( ( (3, 5), (3.5, 5), (4, 4.5) ) ) == (0, 0.5, 0.5) )

    # gen_definitions test
    # 3 domains, constant
    refresult = [(-float('inf'), 1, 2), (1, 2, float('inf'))]
    actualresult = gen_definitions((0, 1, 2, 3), outer_constant=True)
    assert refresult == actualresult

    # 3 domains, not constant
    refresult = [(-float('inf'), 0, 1),
                 (0, 1, 2),
                 (1, 2, 3),
                 (2, 3, float('inf'))]
    actualresult = gen_definitions((0, 1, 2, 3), outer_constant=False)
    assert refresult == actualresult

    # 2 domains, constant
    refresult = [(-float('inf'), 1, float('inf'))]
    actualresult = gen_definitions((0, 1, 2), outer_constant=True)
    assert refresult == actualresult

    # 2 domains, not constant
    refresult = [(-float('inf'), 0, 1),
                 (0, 1, 2),
                 (1, 2, float('inf'))]
    actualresult = gen_definitions((0, 1, 2), outer_constant=False)
    assert refresult == actualresult

    # 1 domain, constant
    refresult = [(-float('inf'), 0, float('inf'))]
    actualresult = gen_definitions((0, 1), outer_constant=True)
    assert refresult == actualresult

    # 1 domain, not constant
    refresult = [(-float('inf'), 0, 1),
                 (0, 1, float('inf'))]
    actualresult = gen_definitions((0, 1), outer_constant=False)
    assert refresult == actualresult

    # partition of unity tests
    pou = partition_of_unity((
        (1, 2, 3),
    ), outer_constant=False)
    assert pou.shape == (3,)

    pou = partition_of_unity((
        (1, 2, 3),
        (4, 5, 6),
    ), outer_constant=False)
    assert pou.shape == (3, 3)
    myfun = pou[1, 1]
    assert myfun((2, 5)) == 1
    assert myfun((2, 5.5)) == 0.5
    assert myfun((2.5, 5.5)) == 0.25
    myfun = pou[1, 2]
    assert myfun((2, 5)) == 0
    assert myfun((2, 6)) == 1

    pou = partition_of_unity((
        (1, 2, 3),
        (4, 5, 6),
    ), outer_constant=True)
    assert pou.shape == (1, 1)
    myfun = pou[0, 0]
    assert myfun((3, 4)) == 1

    pou = partition_of_unity((
        (1, 2),
        (4, 5),
    ), outer_constant=True)
    assert pou.shape == (1, 1)
    myfun = pou[0, 0]
    assert myfun((3, 4)) == 1

    pou = partition_of_unity((
        (1, 2, 3, 5),
        (4, 5, 6, 7),
    ), outer_constant=False)
    assert pou.shape == (4, 4)
    myfun = pou[1, 1]
    assert myfun((2, 5)) == 1



#%% H1 thermal block

from itertools import product
from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction, GenericFunction
from pymor.parameters.functionals import ProjectionParameterFunctional

def thermal_block_problem_h1(num_blocks=(3, 3), parameter_range=(0.1, 1)):
    """Analytical description of a 2D 'thermal block' diffusion problem.

    The problem is to solve the elliptic equation ::

      - ∇ ⋅ [ d(x, μ) ∇ u(x, μ) ] = 1

    on the domain [0,1]^2 with Dirichlet zero boundary values. The domain is
    partitioned into nx x ny blocks and the diffusion function d(x, μ) has
    value μ_i on node i of of the blocks::

                   μ_8      μ_9
      μ_7  *--------*--------*
           |        |        |
           |        |        |
           |        |        |
      μ_4  *--------*--------* μ_6
           |    μ_5 |        |
           |        |        |
           |        |        |
      μ_1  *--------*--------*
                   μ_2      μ_3

    Parameters
    ----------
    num_blocks
        The tuple `(nx, ny)`
    parameter_range
        A tuple `(μ_min, μ_max)`. Each |Parameter| component μ_i is allowed
        to lie in the interval [μ_min, μ_max].
    """

    def parameter_functional_factory(ix, iy):
        return ProjectionParameterFunctional('diffusion',
                                             size=(num_blocks[0]+1)*(num_blocks[1]+1),
                                             index=ix + iy*(num_blocks[0]+1),
                                             name=f'diffusion_{ix}_{iy}')

    pou = localized_pou(num_blocks)

    def diffusion_function_factory(ix, iy):
        pou_func = GenericFunction(pou[ix, iy], dim_domain=2)
        return pou_func

    return StationaryProblem(

        domain=RectDomain(),

        rhs=ConstantFunction(dim_domain=2, value=1.),

        diffusion=LincombFunction([diffusion_function_factory(ix, iy)
                                   for iy, ix in product(range(num_blocks[1]+1), range(num_blocks[0]+1))],
                                  [parameter_functional_factory(ix, iy)
                                   for iy, ix in product(range(num_blocks[1]+1), range(num_blocks[0]+1))],
                                  name='diffusion'),

        parameter_ranges=parameter_range,

        name=f'ThermalBlock({num_blocks})'

    )

#%% field function

from pymor.analyticalproblems.functions import Function

class FieldFunction(Function):
    """Define a 2D |Function| via a constant diffusion field on a particular shape.
    Parameters
    ----------
    bounding_box
        Lower left and upper right coordinates of the domain of the function.
    seed
        random seed for the distribution
    range
        defines the range of the random field
    """

    dim_domain = 2
    shape_range = ()

    def __init__(self, diffusion_field, bounding_box=None):
        bounding_box = bounding_box or [[0., 0.], [1., 1.]]
        n = int(np.sqrt(len(diffusion_field.flatten())))
        shape = (n, n)
        self.diffusion_field = diffusion_field.reshape(shape).T #[:, ::-1]
        self.__auto_init(locals())
        self.lower_left = np.array(bounding_box[0])
        self.size = np.array(bounding_box[1] - self.lower_left)

    def evaluate(self, x, mu=None):
        indices = np.maximum(np.floor((x - self.lower_left) * np.array(self.diffusion_field.shape) /
                                      self.size).astype(int), 0)
        F = self.diffusion_field[np.minimum(indices[..., 0], self.diffusion_field.shape[0] - 1),
                         np.minimum(indices[..., 1], self.diffusion_field.shape[1] - 1)]
        return F
    
#%% struct

class struct():
    pass