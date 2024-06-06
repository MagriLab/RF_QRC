
import numpy as np
from qiskit import  QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from scipy.signal import savgol_filter
from QRC.unitaryblock import Unitary4 , Unitary_FullyEnt , Unitary_FullyEntSym , Unitary_Feature
from qiskit.circuit import ParameterVector
import warnings

class QuantumReservoirNetwork:
    def __init__(self,rho_q,epsilon_q,sigma_in_q,tikh_q,bias_in,bias_out,qubits,N_units,dim,config,emulator,shots,snapshots):
        self.rho_q      = rho_q
        self.epsilon_q  = epsilon_q
        self.sigma_in_q = sigma_in_q
        self.tikh_q     = tikh_q
        self.epsilon_q  = epsilon_q
        self.bias_in    = bias_in
        self.bias_out   = bias_out
        self.qubits     = qubits
        self.N_units    = N_units
        self.dim        = dim
        self.config     = config
        self.emulator   = emulator
        self.shots      = shots
        self.snapshots  = snapshots
        self.alpha      = None
        self.param_qc   = None


    @property
    def rho_q(self):
        return (self._rho_q)

    @rho_q.setter
    def rho_q (self,value):
        self._rho_q = value

    @property
    def tikh_q(self):
        return (self._tikh_q)

    @tikh_q.setter
    def tikh_q (self,value):
        self._tikh_q = value

    @property
    def epsilon_q(self):
        return (self._epsilon_q)

    @epsilon_q.setter
    def epsilon_q (self,value):
        self._epsilon_q = value

    @property
    def qubits(self):
        return (self._qubits)

    @qubits.setter
    def qubits (self,value):
        self._qubits = value

    @property
    def config(self):
        return (self._config)

    @config.setter
    def config (self,value):
        self._config = value

    @property
    def shots(self):
        return (self._shots)

    @shots.setter
    def shots (self,value):
        self._shots = value

    @property
    def emulator(self):
        return (self._emulator)

    @emulator.setter
    def emulator (self,value):
        self._emulator = value

    @property
    def snapshots(self):
        return (self._snapshots)

    @snapshots.setter
    def snapshots (self,value):
        self._snapshots = value

    def gen_random_unitary(self,seed,range):
        rnd = np.random.RandomState(seed)
        alpha = np.zeros(self.qubits)
        alpha_ins = rnd.uniform(0,range,size = (self.qubits)) # Beta as uniform distribution
        alpha     = alpha_ins

        return alpha

    def gen_param_quantumcircuit(self):

        P  = ParameterVector('P'    , self.N_units)
        X  = ParameterVector('X'    , self.dim)
        A  = ParameterVector('alpha', self.qubits)

        q_r = QuantumRegister(self.qubits)
        c   = ClassicalRegister(self.qubits)
        qc  = QuantumCircuit(q_r,c) # Quantum Register , Classical Bits


        if self.config == 1:
            print('Configuration 1')
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,P,qc,'P')
            qc.barrier()

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEntSym(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 2:
            print('Configuration 2')
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,P,qc,'P')
            qc.barrier()

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 3:
            print('Configuration 3')
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 4:
            print('Configuration 4')
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEntSym(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 5:
            print('Configuration 5')
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_Feature(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_Feature(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        self.param_qc = qc

        return self.param_qc

    def load_quantumcircuit(self,prob_p,x_in,alpha):

        # if alpha is not None:
        #     pass
        # else:
        #     raise AttributeError("Random unitary is not defined")

        if self.param_qc is not None:
            pass
        else:
            self.gen_param_quantumcircuit()
            warnings.warn("Parameterized circuit not found, generating...")

        qc = self.param_qc

        if self.config == 1 or self.config == 2:
            #print('With recurrency')
            comb_val =  np.concatenate((prob_p,x_in,alpha)) # Combined Vector of Parameters Values / Chaning Order as Qiskit Parameters are returned Alphabetically

        else:
            #print('Without recurrency')
            comb_val =  np.concatenate((x_in,alpha)) # Combined Vector of Parameters Values / Chaning Order as Qiskit Parameters are returned Alphabetically

        for i , j in enumerate(comb_val):
            #print(i,j)
            bound_qc = qc.assign_parameters({self.param_qc.parameters[i]: j})
            qc  = bound_qc

        self.qc = qc

        return self.qc


    def gen_quantumcircuit(self,prob_p,x_in,alpha):

        P  = prob_p
        X  = x_in
        A  = alpha

        q_r = QuantumRegister(self.qubits)
        c   = ClassicalRegister(self.qubits)
        qc  = QuantumCircuit(q_r,c) # Quantum Register , Classical Bits

        if self.config == 1:
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,P,qc,'P')
            qc.barrier()

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEntSym(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 2:
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,P,qc,'P')
            qc.barrier()

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 3:
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 4:

            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEntSym(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 5:
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_Feature(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_Feature(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        self.qc = qc

        return self.qc


    def method_qc(self,parameterized=False):
        self.parameterized = parameterized
        return self.parameterized

    def quantum_step(self,prob_p,x_in,alpha):

        if self.parameterized:
            self.qc = self.load_quantumcircuit(prob_p,x_in,alpha)
        else:
            self.qc = self.gen_quantumcircuit(prob_p,x_in,alpha)

        if self.emulator   == "sv_sim":

            simulator = Aer.get_backend('aer_simulator_statevector')
            #simulator.set_options(device='GPU')
            self.qc.save_statevector()
            #qc = transpile(qc, simulator)
            result      = simulator.run(self.qc).result()
            statevector = result.get_statevector(self.qc)
            prob_tilde  = np.abs(np.array(statevector)**2)

        elif self.emulator == "qasm_sim":

            self.qc.measure_all(add_bits=False)
            simulator = Aer.get_backend('qasm_simulator')
            simulator = simulator.run(self.qc,shots=self.shots)
            result    = simulator.result()
            counts    = result.get_counts(self.qc)

            a=list(np.zeros(self.N_units))

            for ll in range(self.N_units):
                a[ll]=f'{ll:0{self.qubits}b}'

            # Turning count in terms of probabilities
            psi_tilde = {}
            for output in list(a):
                if output in counts:
                    psi_tilde[output] = counts[output]/self.shots
                else:
                    psi_tilde[output] = 0

            #psi_tilde = dict(sorted(psi_tilde.items())) # sorting dictionary with binary number 0s ---> higher

            psi_tilde = np.array([j for j in psi_tilde.values()]) #Takes values of probabilities from the dictionary

            prob_tilde = (psi_tilde)


        else:
            raise AttributeError("Please select a valid emulator from the list, (a) sv_sim (b) qasm_sim")


        prob_neweps = (1-self.epsilon_q)*prob_p+self.epsilon_q*prob_tilde # including epsilon_q/leaking rate

        # x_in_new = np.dot(Wout_q.T,prob_p) # Prediction Step
        # For next step
        prob_neweps = np.hstack((prob_neweps, self.bias_out)) ###Bias out

        return prob_neweps


    def quantum_openloop(self,U, x0,alpha):
        """ Advances QESN in open-loop.
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
            Xa[i] = self.quantum_step(Xa[i-1,:self.N_units], U[i-1],alpha)

        return Xa


    def quantum_closedloop(self,N,x0,Wout_q,alpha):
        """Advances Quantum Circ in Closed Loop
        Args:
            N : Number of Time Steps
            x0 : Initial Reservoir State
            Wout_q : Output Matrix

        Returns:
            Yh: Time Series of Prediction
            Xa: Final Augmented Reservoir State
        """
        xa    = x0.copy()
        Yh    = np.empty((N+1, self.dim))
        Yh[0] = np.dot(xa, Wout_q)
        for i in np.arange(1,N+1):
            xa    =  self.quantum_step(xa[:self.N_units], Yh[i-1],alpha)
            Yh[i] =  np.dot(xa, Wout_q) #np.linalg.multi_dot([xa, Wout_q])

        return Yh, xa


    def quantum_training(self,U_washout, U_train, Y_train,alpha):
        """ Trains QESN.
            Args:
                U_washout: washout input time series
                U_train: training input time series
                tikh_q: tikhonov factor
            Returns:
                time series of augmented reservoir states
                optimal output matrix
        """

        LHS = 0
        RHS = 0

        N  = U_train[0].shape[0]
        Xa  = np.zeros((U_washout.shape[0], N+1, self.N_units+1))

        for i in range(U_washout.shape[0]): # if training on multiple time-series

            ## washout phase
            xf_washout =  self.quantum_openloop(U_washout[i], np.zeros(self.N_units),alpha)[-1,:self.N_units]
            #xf_washout = quant_open_loop(U_washout[i], np.zeros(N_units_q), sigma_in_q, rho_q,beta,epsilon_q)[-1,:N_units]

            ## open-loop train phase
            Xa[i] = self.quantum_openloop(U_train[i], xf_washout,alpha)
            #Xa[i] = quant_open_loop(U_train[i], xf_washout, sigma_in_q, rho_q,beta,epsilon_q)

            ## Ridge Regression
            LHS  += np.dot(Xa[i,1:].T, Xa[i,1:])
            RHS  += np.dot(Xa[i,1:].T, Y_train[i])

        #solve linear system for each tikh_qonov parameter
        Wout_q = np.zeros((self.tikh_q.size, self.N_units+1,self.dim))
        for j in np.arange(self.tikh_q.size):
            Wout_q[j] = np.linalg.solve(LHS + self.tikh_q[j]*np.eye(self.N_units+1), RHS)
        return Xa, Wout_q, LHS, RHS

    def quantum_openloop_denoised(self,U, x0,alpha,freq,poly):
        """ Advances QESN in open-loop.
            Args:
                U: input time series
                x0: initial reservoir state
            Returns:
                time series of augmented reservoir states
        """
        N     = U.shape[0]
        Xa    = np.empty((N+1, self.N_units+1))
        Xa[0] = np.concatenate((x0,self.bias_out))


        Xa_dn    = np.empty((N+1, self.N_units+1))
        x0_dn    = savgol_filter(x0, window_length = freq, polyorder=poly)
        Xa_dn[0] = np.concatenate((x0_dn,self.bias_out))

        for i in np.arange(1,N+1):
            Xa[i] = self.quantum_step(Xa[i-1,:self.N_units], U[i-1],alpha)

        Xa_dn = Xa_dn.T # reshaping for singal denoise
        for m in range(1,self.N_units+1):
            Xa_dn[m] = savgol_filter(Xa.T[m], window_length = freq, polyorder=poly) # removing output bias before filter
            # Xa_dn[i] = np.concatenate((Xa_dn[m][:-1],self.bias_out))

        Xa_dn = Xa_dn.T # reshaping back for training

        return Xa , Xa_dn

    def quantum_training_denoised(self,U_washout, U_train, Y_train,alpha,freq,poly):
        """ Trains QESN.
            Args:
                U_washout: washout input time series
                U_train: training input time series
                tikh_q: tikh_qonov factor
            Returns:
                time series of augmented reservoir states
                optimal output matrix
        """

        LHS = 0
        RHS = 0

        LHS_dn = 0
        RHS_dn = 0

        N  = U_train[0].shape[0]
        Xa  = np.zeros((U_washout.shape[0], N+1, self.N_units+1))
        Xa_dn  = np.zeros((U_washout.shape[0], N+1, self.N_units+1))

        for i in range(U_washout.shape[0]):

            ## washout phase
            xf_washout , xf_washout_dn =  self.quantum_openloop_denoised(U_washout[i], np.zeros(self.N_units),alpha,freq,poly)

            xf_washout = xf_washout[-1,:self.N_units]
            xf_washout_dn = xf_washout_dn[-1,:self.N_units]


            ## open-loop train phase
            Xa[i] , Xa_dn[i] = self.quantum_openloop_denoised(U_train[i], xf_washout,alpha,freq,poly)
            #Xa[i] = quant_open_loop(U_train[i], xf_washout, sigma_in_q, rho_q,beta,epsilon_q)

            ## Ridge Regression
            LHS  += np.dot(Xa[i,1:].T, Xa[i,1:])
            RHS  += np.dot(Xa[i,1:].T, Y_train[i])

            LHS_dn  += np.dot(Xa_dn[i,1:].T, Xa_dn[i,1:])
            RHS_dn  += np.dot(Xa_dn[i,1:].T, Y_train[i])

        #solve linear system for each tikh_qonov parameter
        Wout_q = np.zeros((self.tikh_q.size, self.N_units+1,self.dim))
        Wout_q_dn = np.zeros((self.tikh_q.size, self.N_units+1,self.dim))


        for j in np.arange(self.tikh_q.size):
            Wout_q[j] = np.linalg.solve(LHS + self.tikh_q[j]*np.eye(self.N_units+1), RHS)
            Wout_q_dn[j] = np.linalg.solve(LHS_dn + self.tikh_q[j]*np.eye(self.N_units+1), RHS_dn)

        return Xa, Wout_q, LHS, RHS , Xa_dn , Wout_q_dn , LHS_dn, RHS_dn
