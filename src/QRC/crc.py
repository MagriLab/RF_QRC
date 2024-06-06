import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

class EchoStateNetwork:
    def __init__(self,tikh,sigma_in,rho,epsilon,bias_in,bias_out,N_units,dim,density):
        self.tikh     = tikh
        self.sigma_in = sigma_in
        self.rho      = rho
        self.epsilon  = epsilon
        self.bias_in  = bias_in
        self.bias_out = bias_out
        self.N_units  = N_units
        self.dim      = dim
        self.density  = density

    # can or cannot use property here, it provides encapsulation
    # @property
    # def tikh(self):
    #     return(self._tikh)

    @property
    def tikh(self):
        return (self._tikh)

    @tikh.setter
    def tikh (self,value):
        self._tikh = value

    @property
    def sigma_in(self):
        return (self._sigma_in)

    @sigma_in.setter
    def sigma_in (self,value):
        self._sigma_in = value

    @property
    def rho(self):
        return (self._rho)

    @rho.setter
    def rho (self,value):
        self._rho = value

    @property
    def epsilon(self):
        return (self._epsilon)

    @epsilon.setter
    def epsilon (self,value):
        self._epsilon = value


    @property
    def norm_u(self):
        return (self._norm_u)

    @norm_u.setter
    def norm_u (self,value):
        self._norm_u = value

    def gen_input_matrix(self,seed):
        rnd = np.random.RandomState(seed)
        #sparse syntax for the input and state matrices
        Win  = lil_matrix((self.N_units,self.dim+1))
        for j in range(self.N_units):
            Win[j,rnd.randint(0, self.dim+1)] = rnd.uniform(-1, 1) #only one element different from zero
        Win = Win.tocsr()

        return Win

    def gen_reservoir_matrix(self,seed):
        rnd = np.random.RandomState(seed)
        W = csr_matrix( #on average only connectivity elements different from zero
        rnd.uniform(-1, 1, (self.N_units, self.N_units)) * (rnd.rand(self.N_units, self.N_units) < (self.density)))

        spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
        W = (1/spectral_radius)*W #scaled to have unitary spec radius

        return W


    def step(self,x_pre, u,Win,W):

        # input is normalized and input bias added

        u_augmented = np.hstack((u/self.norm_u, self.bias_in))

        # reservoir update
        x_post      = (1-self.epsilon)*(x_pre)+self.epsilon*(np.tanh(Win.dot(u_augmented*self.sigma_in) + W.dot(self.rho*x_pre) ))

        #x_post      = (1-self.epsilon)*(x_pre)+self.epsilon*np.tanh(np.dot(u_augmented*self.sigma_in, Win) + self.rho*np.dot(x_pre, W))

        x_augmented = np.hstack((x_post, self.bias_out))

        return x_augmented


    def open_loop(self,U, x0,Win,W):
        """ Advances ESN in open-loop.
            Args:
                U: input time series
                x0: initial reservoir state
            Returns:
                time series of augmented reservoir states
        """
        N     = U.shape[0]
        Xa    = np.empty((N+1, self.N_units+1))
        Xa[0] = np.concatenate((x0,self.bias_out))
        for i in np.arange(1,N+1):
            Xa[i] = self.step(Xa[i-1,:self.N_units], U[i-1],Win,W)

        return Xa

    def closed_loop(self, N, x0, Wout,Win,W):
        """ Advances ESN in closed-loop.
            Args:
                N: number of time steps
                x0: initial reservoir state
                Wout: output matrix
            Returns:
                time series of prediction
                final augmented reservoir state
        """
        xa    = x0.copy()
        Yh    = np.empty((N+1, self.dim))
        Yh[0] = np.dot(xa, Wout)
        for i in np.arange(1,N+1):
            xa    = self.step(xa[:self.N_units], Yh[i-1],Win,W)
            Yh[i] = np.dot(xa, Wout) #np.linalg.multi_dot([xa, Wout])

        return Yh, xa

    def train(self,U_washout, U_train, Y_train,Win,W):
        """ Trains ESN.
            Args:
                U_washout: washout input time series
                U_train: training input time series
                tikh: Tikhonov factor
            Returns:
                time series of augmented reservoir states
                optimal output matrix
        """

        LHS = 0
        RHS = 0

        N  = U_train[0].shape[0]
        Xa  = np.zeros((U_washout.shape[0], N+1, self.N_units+1))

        for i in range(U_washout.shape[0]):

            ## washout phase
            xf_washout = self.open_loop(U_washout[i], np.zeros(self.N_units),Win,W)[-1,:self.N_units]

            ## open-loop train phase
            Xa[i] = self.open_loop(U_train[i], xf_washout,Win,W)

            ## Ridge Regression
            LHS  += np.dot(Xa[i,1:].T, Xa[i,1:])
            RHS  += np.dot(Xa[i,1:].T, Y_train[i])

        #solve linear system for each Tikhonov parameter
        Wout = np.zeros((self.tikh.size, self.N_units+1,self.dim))
        for j in np.arange(self.tikh.size):
            Wout[j] = np.linalg.solve(LHS + self.tikh[j]*np.eye(self.N_units+1), RHS)

        return Xa, Wout, LHS, RHS




    def train_MC(self,U_washout, U_train, Y_train):
        """ Trains ESN.
            Args:
                U_washout: washout input time series
                U_train: training input time series
                Y_train: prediction for next time step to match U_train
                tikh: Tikhonov factor
            Returns:
                Wout: optimal output matrix
        """

        ## washout phase
        # taking last reservoir state of washout [-1] and removing bias :-1
        xf    = self.open_loop(U_washout, np.zeros(self.N_units))[-1,:-1]

        LHS   = 0
        RHS   = 0
        #N_len = (U_train.shape[0]-1)

        ## open-loop train phase
        Xa1 = self.open_loop(U_train, xf)[1:]
        xf  = Xa1[-1,:-1].copy()

        LHS += np.dot(Xa1.T, Xa1)
        RHS += np.dot(Xa1.T, Y_train)

        LHS.ravel()[::LHS.shape[1]+1] += self.tikh

        Wout = np.linalg.solve(LHS, RHS)

        return Wout
