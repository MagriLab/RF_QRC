import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib as mpl
import numpy as np
import math

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
