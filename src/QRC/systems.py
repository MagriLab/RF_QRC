#import torch
import math
import numpy as np
import h5py
#from QESN.solvers import *
from QRC.solvers import Solvers
import scipy


class Systems(Solvers):

    def __init__(self, dt, tot_steps, q0,upsample,N_transient):
        super().__init__(dt, tot_steps, q0,upsample,N_transient)

    def set_param_lorenz63(self):

        self.dim = 3

        self.sigma = 10
        self.r = 28
        self.b = 8/3

        self.t_lyap    = 0.9**(-1)       # Lyapunov Time
        self.N_lyap    = int(self.t_lyap/self.dt)  # number of time steps in one Lyapunov time

        self.N_t = 1 # For generalization, one time series only

        self.N = int(self.tot_steps) #*int(self.t_lyap/self.dt/self.upsample)

        return (self.dim,self.N_lyap,self.N_t)

    def set_param_lorenz96(self,dim):
        self.dim = dim
        if dim==10:
            self.t_lyap    = 1.2**(-1)       # Lyapunov Time dim = 10

        elif dim==20:
            self.t_lyap    = 1.5**(-1)     # Lyapunov Time dim=20

        self.N_lyap    = int(self.t_lyap/self.dt)  # number of time steps in one Lyapunov time

        self.N_t = 1 # For generalization, one time series only

        self.N = int(self.tot_steps) #*int(self.t_lyap/self.dt/self.upsample)

        return (self.dim,self.N_lyap,self.N_t)


    def set_param_MFE(self,N_t):
        self.N_t = N_t
        self.dim = 9
        self.ii  = 0
        self.t_lyap    = 0.0163**(-1)       # Lyapunov Time
        self.N_lyap    = int(self.t_lyap/self.dt)  # number of time steps in one Lyapunov time

        self.N = int(self.tot_steps) #*int(self.t_lyap/self.dt/self.upsample)

        return (self.dim,self.N_lyap)


    def solve_lorenz63(self,u):
        x, y, z = u
        #print(self.sigma,self.r,self.b)
        return np.array([(self.sigma)*(y-x), x*(self.r-z)-y, x*y-self.b*z])


    def solve_lorenz63_jac(self,u):
        x, y, z          = u  # Unpack the state vector
        #Jacobian
        J = np.array([[-self.sigma, self.sigma,     0],
                    [ self.r-z,    -1,    -x],
                    [     y,     x, -self.b]])
        return J


    def solve_lorenz96(self,u):
        p = 8
        return np.roll(u,1) * (np.roll(u,-1) - np.roll(u,2)) - u + p

    def solve_lorenz96_jac(self,u):
        D = len(u)
        J = np.zeros((D,D), dtype='float')
        for i in range(D):
            J[i,(i-1)%D] =  u[(i+1)%D] - u[(i-2)%D]
            J[i,(i+1)%D] =  u[(i-1)%D]
            J[i,(i-2)%D] = -u[(i-1)%D]
            J[i,i] = -1.0
        return J

    def solve_MFE(self,u):

        a1, a2, a3, a4, a5, a6, a7, a8, a9 = u

        # Problem Definition
        Lx = 4*math.pi
        # Ly = 2
        Lz = 2*math.pi
        Re = 400

        # Parameter values
        alfa  = 2*math.pi/Lx
        beta  = math.pi/2
        gamma = 2*math.pi/Lz

        k1 = math.sqrt(alfa**2 + gamma**2)
        k2 = math.sqrt(gamma**2 + beta**2)
        k3 = math.sqrt(alfa**2 + beta**2 + gamma**2)

        dqdt = np.array([beta**2/Re * (1. - a1) - np.sqrt(3/2)*beta*gamma/k3*a6*a8 + np.sqrt(3/2)*beta*gamma/k2*a2*a3,

            - ( 4/3*beta**2 + gamma**2) * a2/Re + 5/3*np.sqrt(2/3)*gamma**2/k1*a4*a6 - gamma**2/np.sqrt(6)/k1*a5*a7 -
            alfa*gamma*beta/np.sqrt(6)/k1/k3*a5*a8 - np.sqrt(3/2)*beta*gamma/k2 * (a1*a3 + a3*a9),

            - (beta**2 + gamma**2)/Re*a3 + 2/np.sqrt(6)*alfa*beta*gamma/k1/k2 * (a4*a7 + a5*a6) +
            (beta**2*(3*alfa**2+gamma**2) - 3*gamma**2*(alfa**2+gamma**2))/np.sqrt(6)/k1/k2/k3*a4*a8,

            - (3*alfa**2 + 4*beta**2)/3/Re*a4 - alfa/np.sqrt(6)*a1*a5 - 10/3/np.sqrt(6)*alfa**2/k1*a2*a6 -
            np.sqrt(3/2)*alfa*beta*gamma/k1/k2*a3*a7 - np.sqrt(3/2)*alfa**2*beta**2/k1/k2/k3*a3*a8 - alfa/np.sqrt(6)*a5*a9,

            - (alfa**2 + beta**2)/Re*a5 + alfa/np.sqrt(6)*a1*a4 + alfa**2/np.sqrt(6)/k1*a2*a7 -
            alfa*beta*gamma/np.sqrt(6)/k1/k3*a2*a8 + alfa/np.sqrt(6)*a4*a9 + 2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a3*a6,

            - (3*alfa**2 + 4*beta**2 + 3*gamma**2)/3/Re*a6 + alfa/np.sqrt(6)*a1*a7 + np.sqrt(3/2)*beta*gamma/k3*a1*a8 +
            10/3/np.sqrt(6)*(alfa**2 - gamma**2)/k1*a2*a4 - 2*np.sqrt(2/3)*alfa*beta*gamma/k1/k2*a3*a5 + alfa/np.sqrt(6)*a7*a9 + np.sqrt(3/2)*beta*gamma/k3*a8*a9,

            - k3**2/Re*a7 - alfa/np.sqrt(6) * (a1*a6 + a6*a9) + (gamma**2 - alfa**2)/np.sqrt(6)/k1*a2*a5 + alfa*beta*gamma/np.sqrt(6)/k1/k2*a3*a4,

            - k3**2/Re*a8 + 2/np.sqrt(6)*alfa*beta*gamma/k3/k1*a2*a5 + gamma**2*(3*alfa**2 - beta**2 + 3*gamma**2)/np.sqrt(6)/k1/k2/k3*a3*a4,

            - 9*beta**2/Re*a9 + np.sqrt(3/2)*beta*gamma/k2*a2*a3 - np.sqrt(3/2)*beta*gamma/k3*a6*a8
            ])

        return dqdt

    def solve_MFE_jac(self,u):
        """
        Defines the jacobian for the differential equations for Moehlis_2004

        Arguments:
            w :  vector of the state variables a_i
            p :  vector of the parameters:
        """
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = u

        # Problem Definition
        Lx = 4*math.pi
        # Ly = 2
        Lz = 2*math.pi
        Re = 400

        # Parameter values
        alfa  = 2*math.pi/Lx
        beta  = math.pi/2
        gamma = 2*math.pi/Lz

        k1 = math.sqrt(alfa**2 + gamma**2)
        k2 = math.sqrt(gamma**2 + beta**2)
        k3 = math.sqrt(alfa**2 + beta**2 + gamma**2)

        J = np.array([[-beta**2/Re, np.sqrt(3/2)*beta*gamma/k2*a3, np.sqrt(3/2)*beta*gamma/k2*a2, 0., 0., - np.sqrt(3/2)*beta*gamma/k3*a8, 0., - np.sqrt(3/2)*beta*gamma/k3*a6, 0.],

                [- np.sqrt(3/2)*beta*gamma/k2*a3, - (4/3*beta**2 + gamma**2)/Re, - np.sqrt(3/2)*beta*gamma/k2*(a1 +a9), 5/3*np.sqrt(2/3)*gamma**2/k1*a6,
                    - gamma**2/np.sqrt(6)/k1*a7 - alfa*gamma*beta/np.sqrt(6)/k1/k3*a8, 5/3*np.sqrt(2/3)*gamma**2/k1*a4, - gamma**2/np.sqrt(6)/k1*a5, - alfa*gamma*beta/np.sqrt(6)/k1/k3*a5, - np.sqrt(3/2)*beta*gamma/k2*a3],

                [0., 0., - (beta**2 + gamma**2)/Re, + 2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a7 + (beta**2*(3*alfa**2+gamma**2) - 3*gamma**2*(alfa**2+gamma**2))/np.sqrt(6)/k1/k2/k3*a8, 2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a6,
                    2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a5, 2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a4, (beta**2*(3*alfa**2+gamma**2) - 3*gamma**2*(alfa**2+gamma**2))/np.sqrt(6)/k1/k2/k3*a4, 0.],

                [- alfa/np.sqrt(6)*a5, - 10/3/np.sqrt(6)*alfa**2/k1*a6, - np.sqrt(3/2)*alfa*beta*gamma/k1/k2*a7 - np.sqrt(3/2)*alfa**2*beta**2/k1/k2/k3*a8, - (3*alfa**2 + 4*beta**2)/3/Re,
                    - alfa/np.sqrt(6)*a1 - alfa/np.sqrt(6)*a9, - 10/3/np.sqrt(6)*alfa**2/k1*a2, - np.sqrt(3/2)*alfa*beta*gamma/k1/k2*a3, - np.sqrt(3/2)*alfa**2*beta**2/k1/k2/k3*a3, - alfa/np.sqrt(6)*a5],

                [alfa/np.sqrt(6)*a4, + alfa**2/np.sqrt(6)/k1*a7 - alfa*beta*gamma/np.sqrt(6)/k1/k3*a8, 2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a6, alfa/np.sqrt(6)*(a1 + a9), - (alfa**2 + beta**2)/Re,
                    2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a3, alfa**2/np.sqrt(6)/k1*a2, - alfa*beta*gamma/np.sqrt(6)/k1/k3*a2, alfa/np.sqrt(6)*a4],

                [alfa/np.sqrt(6)*a7 + np.sqrt(3/2)*beta*gamma/k3*a8, 10/3/np.sqrt(6)*(alfa**2 - gamma**2)/k1*a4, - 2*np.sqrt(2/3)*alfa*beta*gamma/k1/k2*a5, 10/3/np.sqrt(6)*(alfa**2 - gamma**2)/k1*a2,
                    - 2*np.sqrt(2/3)*alfa*beta*gamma/k1/k2*a3, - (3*alfa**2 + 4*beta**2 + 3*gamma**2)/3/Re, alfa/np.sqrt(6)*(a1 + a9), np.sqrt(3/2)*beta*gamma/k3*(a1+a9), alfa/np.sqrt(6)*a7 + np.sqrt(3/2)*beta*gamma/k3*a8],

                [- alfa/np.sqrt(6)*a6, (gamma**2 - alfa**2)/np.sqrt(6)/k1*a5, alfa*beta*gamma/np.sqrt(6)/k1/k2*a4, alfa*beta*gamma/np.sqrt(6)/k1/k2*a3, (gamma**2 - alfa**2)/np.sqrt(6)/k1*a2,
                    - alfa/np.sqrt(6)*(a1 + a9), - k3**2/Re, 0., - alfa/np.sqrt(6)*a6],

                [0., 2/np.sqrt(6)*alfa*beta*gamma/k3/k1*a5, gamma**2*(3*alfa**2 - beta**2 + 3*gamma**2)/np.sqrt(6)/k1/k2/k3*a4, gamma**2*(3*alfa**2 - beta**2 + 3*gamma**2)/np.sqrt(6)/k1/k2/k3*a3,
                    2/np.sqrt(6)*alfa*beta*gamma/k3/k1*a2, 0., 0., - k3**2/Re, 0.],

                [0., np.sqrt(3/2)*beta*gamma/k2*a3, np.sqrt(3/2)*beta*gamma/k2*a2, 0., 0., - np.sqrt(3/2)*beta*gamma/k3*a8, 0., - np.sqrt(3/2)*beta*gamma/k3*a6, - 9*beta**2/Re]
                ])
        return J


    def gen_data_lorenz63(self):
        q         = np.zeros((self.N_t,self.tot_steps-self.N_transient,self.dim))

        for i in range(self.N_t):
            q[i]       = self.solve_ODE(self.solve_lorenz63)

        return(q)

    def gen_data_lorenz96(self):
        q         = np.zeros((self.N_t,self.tot_steps-self.N_transient,self.dim))

        for i in range(self.N_t):
            q[i]       = self.solve_ODE(self.solve_lorenz96)

        return(q)

    def gen_data_MFE(self):
        q         = np.zeros((self.N_t,self.tot_steps-self.N_transient,self.dim))

        for i in range(self.N_t):
            print(i)
            q0         = self.q0
            q0[4]      = self.q0[4] + 0.01*np.random.rand()    #each time series starts from a different perturbed point
            q[i]       = self.solve_ODE(self.solve_MFE)

        print('Laminarized precentage', self.ii/self.N_t)

        ordd = np.nonzero(q[:,-1, 0])
        q    = q[ordd].copy()

        return(q)

    def save_data_MFE(self,q,date):
        fln = './data/'+'MFE'+'_dt='+str(self.dt)+'_N='+str(self.N)+'_dim='+str(self.dim)+'_date='+str(date)
        hf = h5py.File(fln,'w')
        hf.create_dataset('q',data=q)
        hf.close()



    def load_data_MFE(self,date):
        hf      = h5py.File('./data/'+'MFE'+'_dt='+str(self.dt)+'_N='+str(self.N)+'_dim='+str(self.dim)+'_date='+str(date))
        q       = np.array(hf.get('q'))
        hf.close()
        return(q)


    def solve_system_LEs(self,sys, N, Ntherm, u0, dim, norm_time=1):
        """
            Solves the ODEs for N time steps starting from u0.
            Additionally it computes the Lyapunov spectrum and CLVs
            Returned values are normalized.

            Args:
                system: str that defines the dynamical system
                N: number of time steps
                Ntherm: number of time steps for initial transient.
                dt: timestep
                u0: initial condition
                saveclvs: flag to select if to save full CLVs or do a consistency check
                norm_time: modulo of QR decomposition
                params: parameters for ODE
                subspace_LEs_indeces: for high dimensional systems D>3 the ordering of CLVs in subspaces.
            Returns:
                normalized time series of shape (N+1, u0.size)
        """

        print('Dynamical system:',sys)
        if sys=='lorenz63':
            eom = self.solve_lorenz63
            jac = self.solve_lorenz63_jac

        if sys=='lorenz96':
            eom = self.solve_lorenz96
            jac = self.solve_lorenz96_jac

        if sys=='MFE':
            eom = self.solve_MFE
            jac = self.solve_MFE_jac

        #Timesteps in test set
        N_test = N - Ntherm

        T = np.arange(N+1) * self.dt
        Ttest = np.arange(1,int((N_test)/norm_time)+1) * self.dt * norm_time
        xt = u0
        Xt = np.empty((T.size, u0.size))
        Xt[0] = u0

        N_test_norm = int(N_test/norm_time)

        # Lyapunov Exponents timeseries
        LE   = np.zeros((N_test_norm,dim))
        # finite-time Lyapunov Exponents timeseries
        FTLE = np.zeros((N_test_norm,dim))
        # Q matrix recorded in time
        QQ_t = np.zeros((dim,dim,N_test_norm))
        # R matrix recorded in time
        RR_t = np.zeros((dim,dim,N_test_norm))

        #set random orthonormal Lyapunov vectors (GSVs)
        U    = scipy.linalg.orth(np.random.rand(dim,dim))#np.eye(dim)
        Q, R = self.qr_factorization(U)
        U    = Q[:,:dim]

        indx = 0
        for i in range(1, T.size):
            # xt0       = xt
            xn, Mtemp = self.RK4var(self.F_all, eom, jac, xt, U)
            xt       += xn
            U        += Mtemp

            if i%norm_time==0:

                Q, R = self.qr_factorization(U)
                U    = Q[:,:dim]
                if i > Ntherm:
                    QQ_t[:,:,indx] = Q
                    RR_t[:,:,indx] = R
                    LE[indx]       = np.abs(np.diag(R))
                    # Jacobian       = jac(xt, params)
                    FTLE[indx]     = (1./self.dt)*np.log(LE[indx])
                    indx          += 1

            Xt[i] = xt


        LEs = np.cumsum(np.log(LE[:]),axis=0) / np.tile(Ttest[:],(dim,1)).T

        # #Calculation of CLVs
        # if check_clv == 1:
        #     print("Calculate CLVs")
        #     CLV_calculation(system,params,None,None,QQ_t,RR_t,dim,\
        #                     Ntherm,dim,False,dt*norm_time,fname,\
        #                     Xt, None, norm_time, saveclvs,LEs,\
        #                     FTLE, subspace_LEs_indeces)
        # #avFTLES = np.average(FTLE,axis=0)
        # #print("average FTLEs",avFTLES)

        return LEs , Xt
