import numpy as np
from scipy import interpolate
from scipy import linalg
from scipy import optimize

class ConsumptionSavingModel:

    def __init__(self, par):
        self.par = par
        self.sim_m1 = []
        self.data_m1 = []
        pass

    def utility(self,c):
        """ 
        utility   

        
        Args:       
            c (float): consumption            
            rho (float): CRRA parameter       
        
        
        Returns:       
            (float): utility of consumption   
        """
        
        
        return c**(1-self.par.rho)/(1-self.par.rho)

    def bequest(self,m,c):
        """ 
        bequest 

        
        Args:       
            nu (float): bequest motive strength
            m (float): cash-on-hand
            c (float): consumption 
            kappa(float): luxuriousness degree in bequest motive
            rho (float): CRRA parameter       
        
        
        Returns:       
            (float): utility of bequest   
        """
        
        
        return self.par.nu*(m-c+self.par.kappa)**(1-self.par.rho)/(1-self.par.rho)

    def v2(self,c2,m2):
        
        """ 
        value of choice in period 2   
        
        Args:       
            c2 (float): consumption in period 2     
            m2 (float): cash-on-hand in beginning of period 2       
            rho (float): CRRA parameter       
            kappa(float): luxuriousness degree in bequest motive
            rho (float): CRRA parameter 
        
        Returns:       
        
            (float): value-of-choice
        
        """
        
        return self.utility(c2) + self.bequest(m2,c2)

    def v1(self,c1,m1,v2_interp):
        
        
        """ post-decision value function in period 1   
        
        Args:            
            m1 (float): cash-on-hand in the beginning of period 1 
            c1 (float): consumption in period 1
            rho (float): CRRA parameter
            beta (float): discount factor       
            r (float): return on savings       
            Delta (float): income risk scale factor               
            v2_interp (RegularGridInterpolator): interpolator for value function in period 2   
        
        Returns:       
        
            (ndarray): value-of-choice   
        """
        
        # a. v2 value, if low income
        m2_low = (1+self.par.r)*(m1-c1) + 1-self.par.Delta
        v2_low = v2_interp([m2_low])[0]
        
        # b. v2 value, if high income
        m2_high = (1+self.par.r)*(m1-c1) + 1+self.par.Delta
        v2_high = v2_interp([m2_high])[0]
        
        # c. expected v2 value
        expected_v2 = self.par.P_low*v2_low + self.par.P_high*v2_high
        
        # d. total value
        return self.utility(c1) + self.par.beta*expected_v2
        

    def solve_period_2(self):
        
        """ solve consumer problem in period 2    
        
        Args:       
            c2 (float): consumption in period 2     
            m2 (float): cash-on-hand in beginning of period 2       
            rho (float): CRRA parameter       
            kappa(float): luxuriousness degree in bequest motive
            rho (float): CRRA parameter 
        
        
        
        Returns:       
            m2s (ndarray): cash-on-hand in begining of period 2   
            v2s (ndarray): value function      
            c2s (ndarray): consumption function  
        """

        # a. grids
        m2s = np.linspace(1e-4,5,500)
        v2s = np.empty(500)
        c2s = np.empty(500)

        # b. solve for each m2 in grid
        for i,m2 in enumerate(m2s):

            # i. objective
            obj = lambda x: -self.v2(x[0],m2)

            # ii. initial value (consume half)
            x0 = m2/2

            # iii. optimizer
            result = optimize.minimize(obj,[x0],method='L-BFGS-B',bounds=((1e-8,m2),))

            # iv. save
            v2s[i] = -result.fun
            c2s[i] = result.x
            
        return m2s,v2s,c2s
    

    def solve_period_1(self, v2_interp):
        """ post-decision value function in period 1   
              
        
        Args:            
            m1 (float): cash-on-hand in the beginning of period 1 
            c1 (float): consumption in period 1
            rho (float): CRRA parameter
            beta (float): discount factor       
            r (float): return on savings       
            Delta (float): income risk scale factor               
            v2_interp (RegularGridInterpolator): interpolator for value function in period 2  
        
        
        Returns:      
            m1s (ndarray): cash-on-hand in begining of period 1
            v1s (ndarray): value function      
            c1s (ndarray): consumption function 
                                        
         """
         # a. grids
        m1s = np.linspace(1e-8, 4, 100)
        v1s = np.empty(100)
        c1s = np.empty(100)
        
        

        # b. solve for each m1s in grid
        for i, m1 in enumerate(m1s):

            # i. objective
            def obj(x): return -self.v1(x[0], m1, v2_interp)

            # ii. initial guess (consume half)
            x0 = m1/2

            # iii. optimize
            result = optimize.minimize(
                obj, [x0], method='L-BFGS-B', bounds=((1e-12, m1+(1-self.par.Delta)/(1+self.par.r)),))

            # iv. save
            v1s[i] = -result.fun
            c1s[i] = result.x[0]

        return m1s, v1s, c1s
    
    def solve(self):
        
        """
        call and interpolate results
        
        Args:            
            c2 (float): consumption in period 2     
            m2 (float): cash-on-hand in beginning of period 2       
            rho (float): CRRA parameter       
            kappa(float): luxuriousness degree in bequest motive
            beta (float): discount factor       
            r (float): return on savings       
            Delta (float): income risk scale factor  
        
        Returns:   
            m1(ndarray): cash-on-hand in begining of period 1
            v1(ndarray): value function
            c1(ndarray): consumption function 
            m2(ndarray): cash-on-hand in begining of period 2
            v2(ndarray): value function
            c2(ndarray): consumption function 

        """

        # a. solve period 2
        m2, v2, c2 = self.solve_period_2()

        # b. construct interpolator
        v2_interp = interpolate.RegularGridInterpolator([m2], v2,
                                                        bounds_error=False, fill_value=None)

        # b. solve period 1
        m1, v1, c1 = self.solve_period_1(v2_interp)

        return m1, v1, c1, m2, v2, c2
    
    def simulate(self):
        
        """
        solves and interpolates simulated values
        
        Args:            
            sim_m1 (float): simulated values of cash-on-hand in begining of period 1
            c2 (float): consumption in period 2     
            m2 (float): cash-on-hand in beginning of period 2       
            rho (float): CRRA parameter       
            kappa(float): luxuriousness degree in bequest motive
            beta (float): discount factor       
            r (float): return on savings       
            Delta (float): income risk scale factor  
        
        Returns:   
            sim_c1(ndarray): consumption in period 1

        """

        # a. solve the model at current parameters
        m1, v1, c1, m2, v2, c2 = self.solve()


        # b. construct interpolaters
        c1_interp1 = interpolate.RegularGridInterpolator([m1], c1,
                                                        bounds_error=False, fill_value=None)

    
        # c. sim period 1 based on draws of initial m and solution
        sim_c1 = c1_interp1(self.sim_m1)



        return sim_c1

