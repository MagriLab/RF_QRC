# Solvers for solving chaotic system of equations
#import torch
# from QRC.systems import *
import numpy as np


class Solvers:
    def __init__(self,dt,tot_steps,q0,upsample,N_transient):
        self.dt          = dt
        self.tot_steps   = tot_steps
        self.q0          = q0
        self.upsample    = upsample
        self.N_transient = N_transient

    def RK4(self,func):
        """4th order RK for autonomous systems described by func

        Args:
            func (_type_): system of equations to be solved

        Returns:
            q: Time series of the system
        """
        if self.t_lyap == 0.0163**(-1) : # The lyapunov time of MFE
            global ii #allows to use the variable outside the function as well
            ii= self.ii

        q        = np.zeros((self.tot_steps+1,self.q0.shape[0]))
        q[0]     = self.q0
        k        = np.zeros(self.tot_steps+1)

        for i in 1+np.arange(self.tot_steps):

            k1   = self.dt * func(q[i-1])
            k2   = self.dt * func(q[i-1] + k1/2)
            k3   = self.dt * func(q[i-1] + k2/2)
            k4   = self.dt * func(q[i-1] + k3)

            q[i] = q[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6

            if self.t_lyap == 0.0163**(-1) : # The lyapunov time of MFE

                k[i] = 0.5*np.linalg.norm(q[i,:self.dim])**2

                #check laminarization
                if k[i] > 0.48 and i > self.N_transient:
                    print('Laminarized', self.ii)
                    self.ii += 1
                    q[i,-1] = 100
                    break
        return  q

    def solve_ODE(self,func):

        ''' Compute time series '''

        q   = self.RK4(func)

        return q[self.N_transient+1:] # only return the data after removing transients



    def F_all(self,eom, jac, u, M):

        ''' solves system. u is the state vector composed
            by the variables. J is the Jacobian and M are the
            Gram-Schmidt Lyapunov vectors'''

        dqdt  = eom(u)
        J     = jac(u)
        dMdt  = np.dot(J, M)

        return dqdt, dMdt


    def RK4var(self,f, eom, jac, x, mat):  # (F_all, eom, jac, xt, dt, params, U)

        K1, M1 = f(eom, jac, x, mat)
        K2, M2 = f(eom, jac, x + self.dt*K1/2.0, mat + self.dt*M1/2.0)
        K3, M3 = f(eom, jac, x + self.dt*K2/2.0, mat + self.dt*M2/2.0)
        K4, M4 = f(eom, jac, x + self.dt*K3, mat + self.dt*M3)

        A = np.array(self.dt * (K1/2.0 + K2 + K3 + K4/2.0) / 3.0)
        B = np.array(self.dt * (M1/2.0 + M2 + M3 + M4/2.0) / 3.0)

        return A, B

    def qr_factorization(self,A):
        m, n = A.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))

        for j in range(n):
            v = A[:, j].copy()

            for i in range(j):
                q = Q[:, i]
                R[i, j] = q.dot(v)
                v = v - R[i, j] * q

            norm = np.linalg.norm(v)
            Q[:, j] = v / norm
            R[j, j] = norm
        return Q, R


def RK4(q0,dt,N,func,N_transient):
    """4th order RK for autonomous systems described by func

    Args:
        q0 (_type_): initial conditions
        dt (_type_): time step
        N (_type_): length of time series
        func (_type_): system of equations to be solved

    Returns:
        q: Time series of the system
    """
    global ii #allows to use the variable outside the function as well
    ii=0
    funct = str(func)
    #q        = torch.zeros((N+1,q0.shape[0]))
    q        = np.zeros((N+1,q0.shape[0]))
    q[0]     = q0
    k        = np.zeros(N+1)

    #for i in 1+torch.arange(N):
    for i in 1+np.arange(N):
        k1   = dt * func(q[i-1])
        k2   = dt * func(q[i-1] + k1/2)
        k3   = dt * func(q[i-1] + k2/2)
        k4   = dt * func(q[i-1] + k3)

        q[i] = q[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6

        if funct == 'MFE':
            dim = 9 # using 9 modes
            #k[i] = 0.5*torch.linalg.norm(q[i,:Ndim])**2
            k[i] = 0.5*np.linalg.norm(q[i,:dim])**2


            #check laminarization
            if k[i] > 0.48 and i > N_transient:
                print('Laminarized', ii)
                ii += 1
                q[i,-1] = 100
                break


    return  q


def solve_ODE(dt,N,N_transient,q0,func):

    ''' Compute MFE time series '''

    q   = RK4(q0,dt,N,func,N_transient)

    return q[N_transient+1:] # only return the data after removing transients
