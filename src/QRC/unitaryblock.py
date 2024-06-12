



import numpy as np
from itertools import combinations
#%matplotlib inline

def Unitary1(n,k,qc,prob_p,name='param'):
    """_summary_

    Args:
        n - number of qubits
        k - index for probabilities and for repeating circuits until Nres [range(0,(2**n)-1,n)]
        q - quantum register
        c - classical register
        qc- quantum circuit
        prob_p - Probabilties for qubit gate
        gate1 - first single qubit gate
        gate2 - second multi qubit gate

    Returns:
        qc: Qubit Circuit with Unitary 1
    """

    Nres = 2**n
    for i in range(n):
            if k+i >= Nres:
                break

            if i <= 1:
                qc.ry(((prob_p[i+k])),i, label=f'$R_Y$({name})')
                qc.cx(i,i+1)

        #print(i)
            if i > 1 :
                if i % (n-1) == 0:
                    #print('Mod',i)
                    qc.ry(((prob_p[i+k])),i, label=f'$R_Y$({name})')
                    qc.cx(i,i-1)
                else:
                    qc.ry(((prob_p[i])),i, label=f'$R_Y$({name})')
                    qc.cx(i,i+1)
    qc.barrier()
    return qc


def Unitary2(X,qc,name='param'):
    """_summary_

    Args:
        X - array containing initial conditions of the systems
        n_in - how many inputs to take
        qc - initialized circuit
    Returns:
        qc: Qubit Circuit for Unitary 2
    """
     #A1 A2 A3 B1, B2 ... B4
    #X = X[:n_in] #taking 3 inputs
    #qc.append(Unitary(), [0,1,2,3,4,5,6,7,8])
    for i in range(len(X)):
        qc.ry(((X[i])),i, label=f'$R_Y$({name})')
        if i < len(X)-1:
            qc.cx(i,i+1)

    qc.barrier()
    return qc


def Unitary3(n,beta,qc,name='param'):
    """_summary_
    Args:
        q - quantum register
        c - classical register
        n - number of qubits
        beta - random rotation angles
        qc - initialized circuit
    Returns:
        qc: Qubit Circuit for Unitary 3
    """
    for i in range(n):
        if i <= 1:
            qc.ry(((beta[i])),i, label=f'$R_Y$({name})')
            qc.cx(i,i+1)

        #print(i)
        if i > 1 :
            if i % (n-1) == 0:
                #print('Mod',i)
                qc.ry(((beta[i])),i, label=f'$R_Y$({name})')
                qc.cx(i,i-1)
            else:
                qc.ry(((beta[i])),i, label=f'$R_Y$({name})')
                qc.cx(i,i+1)
    qc.barrier()
    return qc


def Unitary4(n,X,qc,name='param'):
    """_summary_
        A generalized Unitary Function for Linearly Entangled Qubits
    Args:
        q - quantum register
        c - classical register
        n - number of qubits
        X - Input
        qc - initialized circuit
    Returns:
        qc: Qubit Circuit
    """
    for j , param in enumerate(X):
        i = j % n
        qc.ry(param,i, label=f'$R_Y$({name})')

        if i <= 1:
            qc.cx(i,i+1)
        #print(i)
        if i > 1 :
            if i % (n-1) == 0:
                #print('Mod',i)
                qc.cx(i,i-1)
            else:
                qc.cx(i,i+1)
        if i == n-1:
            qc.barrier()
    return qc


def Unitary_Linear(n,X,qc,name='param'):
    """_summary_
        A generalized Unitary Function for Linearly Entangled Qubits
    Args:
        q - quantum register
        c - classical register
        n - number of qubits
        X - Input
        qc - initialized circuit
    Returns:
        qc: Qubit Circuit
    """
    for j , param in enumerate(X):
        i = j % n
        qc.ry(param,i)

    for j , param in enumerate(X):
        i = j % n
        if i % (n-1) == 0 and i > 1:
            #print('Mod',i)
            qc.cx(i,0)
        else:
            qc.cx(i,i+1)


    return qc


def Unitary_ReverseEnt(n,X,qc,name='param'):
    """_summary_
        A generalized Unitary Function for Linearly Entangled Qubits
    Args:
        q - quantum register
        c - classical register
        n - number of qubits
        X - Input
        qc - initialized circuit
    Returns:
        qc: Qubit Circuit
    """

    for k in range(n-1):
            qc.cx(k,k+1)

    for j , param in enumerate(X):
        i = j % n
        qc.ry(param,i)

    for k in reversed(range(n-1)):
        qc.cx(k,k+1)

    # for k,l in reversed(enumerate(X[:-1])):
    #     qc.cx(k,k+1)

        # i = j % n
        # print(i,j)
        # if i % (n-1) == 0 and i > 1:
        #     #print('Mod',i)
        #     qc.cx(i,0)
        # else:
        #     qc.cx(i,i+1)

    qc.barrier()
    return qc

def Unitary_FullyEnt(n,X,qc,name='param'):
    """_summary_
        A generalized Unitary Function for Linearly Entangled Qubits
    Args:
        q - quantum register
        c - classical register
        n - number of qubits
        X - Input
        qc - initialized circuit
    Returns:
        qc: Qubit Circuit
    """
    for j , param in enumerate(X):
        if j < n:
            i = j % n
            qc.ry(param,i,label=f'$R_Y$({name})')

    comb = combinations(range(n), 2)

    for ll in (list(comb)):
        qc.cx(ll[0],ll[1])

    # qc.barrier()

    for j , param in enumerate(X):
        if j >= n:
            i = j % n
            qc.ry(param,i,label=f'$R_Y$({name})')

    # qc.barrier()

    # ADD ONE MORE ALL TO ALL LAYER? only for j > n

    return qc

def Unitary_Feature(n,X,qc,name='param'):
    """_summary_
        A generalized Unitary Function
    Args:
        q - quantum register
        c - classical register
        n - number of qubits
        X - Input
        qc - initialized circuit
    Returns:
        qc: Qubit Circuit
    """
    for j , param in enumerate(X):
        i = j % n
        #qc.ry(param,i, label=f'$R_Y$({name})')
        qc.rz(param,i)

    comb = combinations(range(n), 2)

    if len(X) < n: # Incase input is not equal to number of qubits
        comb = combinations(range(len(X)), 2)

    for ll in (list(comb)):
        qc.cx(ll[0],ll[1])
        qc.ry(X[ll[0]]*X[ll[1]],ll[1])
        qc.cx(ll[0],ll[1])

    # for l in reversed(list(comb)):
    #     qc.cx(l[0],l[1])
    #qc.barrier()
    return qc

def Unitary_FullyEntSym(n,X,qc,name='param'):
    """_summary_
        A generalized Unitary Function for Fully Entangled Qubits
    Args:
        q - quantum register
        c - classical register
        n - number of qubits
        X - Input
        qc - initialized circuit
    Returns:
        qc: Qubit Circuit
    """
    for j , param in enumerate(X):
        i = j % n
        qc.ry(param,i, label=f'$R_Y$({name})')

    comb = combinations(range(n), 2)

    for ll in (list(comb)):
        qc.cx(ll[0],ll[1])

    #qc.barrier()
    for j , param in enumerate(X):
        i = j % n
        qc.ry(param,i, label=f'$R_Y$({name})')
    # for l in reversed(list(comb)):
    #     qc.cx(l[0],l[1])

    return qc

def Unitary_C(n,X,qc,name='param'):
    """_summary_
        A generalized Unitary Function for Linearly Entangled Qubits
    Args:
        q - quantum register
        c - classical register
        n - number of qubits
        X - Input
        qc - initialized circuit
    Returns:
        qc: Qubit Circuit
    """
    for j , param in enumerate(X):

        i = j % n
        qc.ry(param,i, label=f'$R_Y$({name})')

        if i > 0:
            qc.cx(i,i-1)
        if i < n-1:
            qc.cx(i,i+1)

        # if i % (n-1) == 0:
        #     qc.cx(i,i-1)
        # else:
        #     qc.cx(i,i+1)

        if i == n-1:
            qc.barrier()
    return qc
