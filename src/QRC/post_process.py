import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib as mpl
import numpy as np


def add_noise(U,target_snr_db,seed,dim,N0,N1):
    """_summary_

    Args:
        U (_flattened_array_): _2dim_Input_Data(Flattened)

    Returns:
        _UU_: _3dim_Reshaped_array
    """
    #### adding noise component-wise to the data

    # Set a target SNR in decibel
    #target_snr_db = 40
    sig_avg_watts = np.var(U,axis=0) #signal power
    sig_avg_db = 10 * np.log10(sig_avg_watts) #convert in decibel
    # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.zeros(U.shape)
    seed = 0                        #to be able to recreate the data
    rnd  = np.random.RandomState(seed)
    for i in range(dim):
        noise_volts[:,i] = rnd.normal(mean_noise, np.sqrt(noise_avg_watts[i]),
                                        U.shape[0])
    UU  = U + noise_volts
    UU  = UU.reshape(N0,N1,dim)

    return UU

def plot_lorenz63_attractor(U,length):
    """A function to plot input data of Lorenz 63

    Args:
        U (array): Input Time Series
        length (scalar): length of Plot 1

    Return:
        Plot 1 : 3d Plot of x
        Plot 2 : Time Series wrt Number of Steps
        Plot 3 : Time Series wrt Lyapunov Time
        Plot 4 : Convection Current and Thermal Plots
    """

    # 3D PLOT OF LORENZ 63 ATTRACTOR

    plt.rcParams['text.usetex'] = True
    plt.rcParams["figure.figsize"] = (10,6)
    plt.rcParams["font.size"] = 12
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlabel("$x_{1}$",labelpad=5)
    ax.set_ylabel("$x_{2}$",labelpad=5)
    ax.set_zlabel("$x_{3}$",labelpad=5)

    plt.tight_layout()

    ax.plot(*U[:length,:].T, lw=0.6, c ='blue')
    #ax.scatter(*U[10:11,:].T, c ='red')
    ax.dist = 11.5
    plt.legend(['Training Data'])


def plot_lorenz63_time(U,N_lyap,l2,l3):
    """A function to plot input data of Lorenz 63

    Args:
        U (array): Input Time Series
        l1 (scalar): length of Plot 1
        l2 (scalar): length of Plot 2
        l3 (scalar): length of Plot 3

    Return:
        Plot 1 : 3d Plot of Attractor
        Plot 2 : Time Series wrt Number of Steps
        Plot 3 : Time Series wrt Lyapunov Time
        Plot 4 : Convection Current and Thermal Plots
    """

    # PLOTTING TIME EVOLUTION OF A1,B1,B2 in time steps

    t_len = l2 # length of time series to plot
    t_str = 0 # starting points for plots
    #### 222 time steps in one Lyapunov Time, have to do it by 3Lambda to replicate the paper
    fig, axs = plt.subplots(3)
    fig.suptitle('Time steps Vs $x_{1}$,$x_{2}$,$x_{3}$')
    axs[0].plot(np.arange(t_str,t_len),U[t_str:t_len,0])
    axs[0].set_ylabel("$x_{1}$")
    axs[1].plot(np.arange(t_str,t_len),U[t_str:t_len,1])
    axs[1].set_ylabel("$x_{2}$")
    axs[2].plot(np.arange(t_str, t_len),U[t_str:t_len,2])
    axs[2].set_xlabel('Time steps')
    axs[2].set_ylabel("$x_{3}$")

    # PLOTTING TIME EVOLUTION OF A1,B1,B2 in LYAP TIME
    # Readjusting Lyap Time to zero

    t_len = l3 # length of time series to plot
    t_str = 0 # starting points for plots, because have removed transients already
    #### 222 time steps in one Lyapunov Time, have to do it by 3Lambda to replicate the paper
    fig, axs = plt.subplots(3)
    fig.suptitle('Lyapunov Time Vs $x_{1}$,$x_{2}$,$x_{3}$')
    axs[0].plot(np.arange(t_str,t_len)/N_lyap,U[t_str:t_len,0])
    axs[0].set_ylabel("$x_{1}$")
    axs[1].plot(np.arange(t_str,t_len)/N_lyap,U[t_str:t_len,1])
    axs[1].set_ylabel("$x_{2}$")
    axs[2].plot(np.arange(t_str, t_len)/N_lyap,U[t_str:t_len,2])
    axs[2].set_xlabel('Lyapunov Time (LT)')
    axs[2].set_ylabel("$x_{3}$")


def lorenz63_timeseries_plot(N_test,N_tstart,N_intt,N_fwd,j,UU,U,U_test,N_washout,N_lyap,ensemble,N_t,minimum,esn,Win,W,QESN,Woutt,Woutt_qq,alpha_qq,plot=True,quantum=True):

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(UU)))
    threshold_ph = 0.5
    num_series = 0
    n_plot = 3

    PH_plot   = np.zeros((ensemble,3))
    PH_plot_q = np.zeros((ensemble,3))

    print(U.shape, UU.shape, N_t)


    print('Realization    :',j+1)
    #load matrices and hyperparameters
    Wout     = Woutt[j]
    # Win      = Winn[j]
    # W        = Ws[j]
    esn.rho      = 10**minimum[j,0]
    esn.epsilon  = minimum[j,1]
    esn.sigma_in = minimum[j,2]

    if quantum:
        Wout_q             = Woutt_qq[j]
        alpha              = alpha_qq[j]

        print('Hyperparameters:',esn.rho, esn.epsilon,esn.sigma_in,esn.tikh)
        print('Quantum Hyperparameters:', QESN.rho_q, QESN.epsilon_q,QESN.sigma_in_q,QESN.tikh_q,'Seed:',j+1)

        # to store prediction horizon in the test set
        PH         = np.zeros(N_test)
        PH_q       = np.zeros(N_test)
        # to plot results

        if plot:
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout(h_pad=12)

        #run different test intervals
        for i in range(N_test):

            # data for washout and target in each interval
            U_wash    = U_test[num_series,N_tstart - N_washout +i*N_fwd : N_tstart + i*N_fwd].copy()
            Y_t       = U_test[num_series,N_tstart  +i*N_fwd           : N_tstart + i*N_fwd + N_intt].copy()

            #washout for each interval
            Xa1     = esn.open_loop(U_wash, np.zeros(esn.N_units),Win,W)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t        = esn.closed_loop(N_intt-1, Xa1[-1], Wout,Win,W)[0]
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph:
                PH[i] = N_intt/N_lyap #(in case PH is larger than interval)


            if quantum:
                #washout for each interval
                Xa1_q   = QESN.quantum_openloop(U_wash, np.zeros(QESN.N_units),alpha) # here concatenation and bias out addition

                Uh_wash_q = np.dot(Xa1_q, Wout_q)

                # Prediction Horizon
                Yh_t_q        = QESN.quantum_closedloop(N_intt-1, Xa1_q[-1], Wout_q,alpha)[0]
                Y_err_q       = np.sqrt(np.mean((Y_t-Yh_t_q)**2,axis=1))/sigma_ph
                PH_q[i]       = np.argmax(Y_err_q>threshold_ph)/N_lyap
                if PH_q[i] == 0 and Y_err_q[0]<threshold_ph:
                    PH_q[i] = N_intt/N_lyap #(in case PH is larger than interval)


            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    plt.subplot(n_plot,2,1+i*2)
                    xx = np.arange(U_wash[:,0].shape[0])/N_lyap

                    plt.plot(xx,U_wash[:U_wash.shape[0],i], 'k',label='True')
                    plt.plot(xx,Uh_wash[:U_wash.shape[0],i], '--r',label='ESN')
                    if quantum:
                        plt.plot(xx,Uh_wash_q[:U_wash.shape[0],i], '--b',label='QESN')

                    plt.xlabel('Time[Lyapunov Times]')
                    plt.ylabel('$x$'+str(i+1))
                    if i==0:
                        plt.legend(ncol=2)
                        plt.title('Washout Phase')

                    plt.subplot(n_plot,2,2+i*2)
                    plt.axvline(PH[i])
                    xx = np.arange(Y_t[:,0].shape[0])/N_lyap
                    plt.plot(xx,Y_t[:,i], 'k')
                    plt.plot(xx,Yh_t[:,i], '--r')
                    if quantum:
                        plt.plot(xx,Yh_t_q[:,i], '--b',label='QESN')
                    if i==0:
                        plt.legend(ncol=2)
                        plt.title('Testing Phase')

                    plt.xlabel('Time [Lyapunov Times]')

        # Percentiles of the prediction horizon
        PH_plot[j] = [np.quantile(PH,.75), np.median(PH), np.quantile(PH,.25)]
        print('PH quantiles [Lyapunov Times]:',
            PH_plot[j])
        print('')

        if quantum:
            # Percentiles of the prediction horizon
            PH_plot_q[j] = [np.quantile(PH_q,.75), np.median(PH_q), np.quantile(PH_q,.25)]
            print('PH quantiles [Lyapunov Times]:',
                PH_plot_q[j])
            print('')
        plt.show()

    return U_wash, Uh_wash, Y_t , Yh_t , PH_plot , Uh_wash_q, Yh_t_q , PH_plot_q


def MFE_timeseries_plot(N_test,N_tstart,N_intt,N_fwd,j,UU,U,U_test,N_washout,N_lyap,ensemble,N_t,minimum,esn,Win,W,QESN,Woutt,Woutt_qq,alpha_qq,plot=True,quantum=True):

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(UU)))
    threshold_ph = 0.5
    num_series = 0
    n_plot = 9

    PH_plot   = np.zeros((ensemble,3))
    PH_plot_q = np.zeros((ensemble,3))

    print(U.shape, UU.shape, N_t)


    print('Realization    :',j+1)
    #load matrices and hyperparameters
    Wout     = Woutt[j]
    # Win      = Winn[j]
    # W        = Ws[j]
    esn.rho      = 10**minimum[j,0]
    esn.epsilon  = minimum[j,1]
    esn.sigma_in = minimum[j,2]

    if quantum:
        Wout_q             = Woutt_qq[j]
        alpha              = alpha_qq[j]

        print('Hyperparameters:',esn.rho, esn.epsilon,esn.sigma_in,esn.tikh)
        print('Quantum Hyperparameters:', QESN.rho_q, QESN.epsilon_q,QESN.sigma_in_q,QESN.tikh_q,'Seed:',j+1)

        # to store prediction horizon in the test set
        PH         = np.zeros(N_test)
        PH_q       = np.zeros(N_test)
        # to plot results

        if plot:
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout(h_pad=12)

        #run different test intervals
        for i in range(N_test):

            # data for washout and target in each interval
            U_wash    = U_test[num_series,N_tstart - N_washout +i*N_fwd : N_tstart + i*N_fwd].copy()
            Y_t       = U_test[num_series,N_tstart  +i*N_fwd           : N_tstart + i*N_fwd + N_intt].copy()

            #washout for each interval
            Xa1     = esn.open_loop(U_wash, np.zeros(esn.N_units),Win,W)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t        = esn.closed_loop(N_intt-1, Xa1[-1], Wout,Win,W)[0]
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph:
                PH[i] = N_intt/N_lyap #(in case PH is larger than interval)


            if quantum:
                #washout for each interval
                Xa1_q   = QESN.quantum_openloop(U_wash, np.zeros(QESN.N_units),alpha) # here concatenation and bias out addition

                Uh_wash_q = np.dot(Xa1_q, Wout_q)

                # Prediction Horizon
                Yh_t_q        = QESN.quantum_closedloop(N_intt-1, Xa1_q[-1], Wout_q,alpha)[0]
                Y_err_q       = np.sqrt(np.mean((Y_t-Yh_t_q)**2,axis=1))/sigma_ph
                PH_q[i]       = np.argmax(Y_err_q>threshold_ph)/N_lyap
                if PH_q[i] == 0 and Y_err_q[0]<threshold_ph:
                    PH_q[i] = N_intt/N_lyap #(in case PH is larger than interval)


            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    plt.subplot(n_plot,2,1+i*2)
                    xx = np.arange(U_wash[:,0].shape[0])/N_lyap

                    plt.plot(xx,U_wash[:U_wash.shape[0],i], 'k',label='True')
                    plt.plot(xx,Uh_wash[:U_wash.shape[0],i], '--r',label='ESN')
                    if quantum:
                        plt.plot(xx,Uh_wash_q[:U_wash.shape[0],i], '--b',label='QESN')

                    plt.xlabel('Time[Lyapunov Times]')
                    plt.ylabel('$x$'+str(i+1))
                    if i==0:
                        plt.legend(ncol=2)
                        plt.title('Washout Phase')

                    plt.subplot(n_plot,2,2+i*2)
                    plt.axvline(PH[i])
                    xx = np.arange(Y_t[:,0].shape[0])/N_lyap
                    plt.plot(xx,Y_t[:,i], 'k')
                    plt.plot(xx,Yh_t[:,i], '--r')
                    if quantum:
                        plt.plot(xx,Yh_t_q[:,i], '--b',label='QESN')
                    if i==0:
                        plt.legend(ncol=2)
                        plt.title('Testing Phase')

                    plt.xlabel('Time [Lyapunov Times]')

        # Percentiles of the prediction horizon
        PH_plot[j] = [np.quantile(PH,.75), np.median(PH), np.quantile(PH,.25)]
        print('PH quantiles [Lyapunov Times]:',
            PH_plot[j])
        print('')

        if quantum:
            # Percentiles of the prediction horizon
            PH_plot_q[j] = [np.quantile(PH_q,.75), np.median(PH_q), np.quantile(PH_q,.25)]
            print('PH quantiles [Lyapunov Times]:',
                PH_plot_q[j])
            print('')
        plt.show()

    return U_wash, Uh_wash, Y_t , Yh_t , PH_plot , Uh_wash_q, Yh_t_q , PH_plot_q
