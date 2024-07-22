



import numpy as np
from itertools import combinations
#%matplotlib inline


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


def Unitary_FullyEnt(n,X,qc,name='param'):
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
        A generalized Unitary Function for feature map
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
