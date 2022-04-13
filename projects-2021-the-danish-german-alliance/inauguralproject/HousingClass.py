####################################################################
#This file contains functions for the individual utility, individual 
#tax revenue, population average tax revenue, and the solution to the 
#household problem, as well the HousingClass with innitial values of 
#parameters and the methods corresponding to the functions.
####################################################################

import numpy as np
from scipy import optimize


def u_func(model,c,p_h):
    """ class utility function returns utility for given 
    consumption level and housing quality stated as housing price
    
    Args:
        
        model: class attributes
        c: consumption
        p_h: housing quality stated as housing price
         
    Returns:
    
        utility
    
    """
    return c**(1-model.phi)*p_h**model.phi 


def t_func(model,p_h):
    """class tax function calculates total tax revenue 
    for given housing price p_h
  
    Args:

        model: class attributes
        p_h: housing price
   
   Returns:
    
        individual tax revenue
        
    """
    return model.tau_g*(p_h*model.epsilon)+model.tau_p*max((p_h*model.epsilon)-model.p_bar,0)


def t_avrg_func(model, N):
    """Class average tax function returns average tax revenue for
    N random draws of cash in hand from lognormal distribution
    
    Args:

        model: class attributes
        N: number of draws from lognormal distribution
   
   Returns:
    
        average tax revenue across N individuals
        model.M: saves draws from distribution
        model.p_h_list: saves list of optimal housing prices for M draws
        
    """
    np.random.seed(1)
    M = np.random.lognormal(mean=-0.4, sigma=0.35, size=N)
    
    p_h_list = []
    t_list = []
    for m in M:
        model.m = m
        model.solve()
        t_list.append(model.t)
        p_h_list.append(model.p_h)
   
    model.t_avrg = sum(t_list)/N
    
    # for distribution plots M and h
    model.M = M
    model.p_h_list = p_h_list 
    
    return model.t_avrg


def solve(model):
    """solve function minimzes negative utility and saves 
    optimal values of consumption, housing, and corresponding tax revenue   
    
    Args:
        
        model: class attributes

    Returns (saves):
    
        model.c: optimal consumption of class
        model.p_h: optimal housing quality stated as house price of class
        model.u: utility for optimal consumption and housing
        model.t: tax revenue corresponding to optimal consumption and housing of class
    
    """
    # a. objective function (to minimize) 
    obj = lambda x: -model.u_func(x[0],x[1]) # minimize -> negtive of utility
        
    # b. constraints and bounds
    budget_constraint = lambda x: model.m-x[0]-(model.r*x[1]+model.tau_g*(x[1]*model.epsilon)+model.tau_p*max((x[1]*model.epsilon)-model.p_bar,0)) # violated if negative
    constraints = ({'type':'ineq','fun':budget_constraint})
        
    # c. call solver
    x0 = [model.m/2,model.m/2]    #initial guess
    sol = optimize.minimize(obj,x0,method='SLSQP',constraints=constraints)
        
    # d. save
    model.c = sol.x[0]
    model.p_h = sol.x[1]
    model.u = model.u_func(model.c,model.p_h)
    model.t = model.t_func(model.p_h)
    
    
class HousingClass:
    """Creates class for household problem, 
    initiates default attributes as stated in question 1,
    includes functions defined above """

    def __init__(self):
        
        # set parameter values
        self.m = 0.5
        self.phi = 0.3
        self.epsilon = 0.5
        self.r = 0.03
        self.tau_g = 0.012
        self.tau_p = 0.004
        self.p_bar = 3
            
    solve = solve
    u_func = u_func
    t_func = t_func
    t_avrg_func = t_avrg_func