import numpy as np
from math import gcd
# from scipy.optimize import linprog
import matplotlib.pyplot as plt

import ot
import ot.plot
import matplotlib.pylab as pl

def OT_solver(P,Q,m,n,fig_1=True,fig_3=True,fig_4=True):
    """
    Using Python OT solver:
    https://pythonot.github.io/auto_examples/plot_OT_2D_samples.html#sphx-glr-auto-examples-plot-ot-2d-samples-py
    """
    M = ot.dist(P,Q, p=1)
    a,b = np.ones((n,)) / n, np.ones((m,)) / m
    
    if fig_1:
        pl.figure(1)
        pl.plot(P[:, 0], P[:, 1], '+b', label='Source samples')
        pl.plot(Q[:, 0], Q[:, 1], 'xr', label='Target samples')
        pl.legend(loc=0)
        pl.title('Source and target distributions')

        pl.figure(2)
        pl.imshow(M, interpolation='nearest')
        pl.title('Cost matrix M')
    G0 = ot.emd(a, b, M)
    if fig_3:
        pl.figure(3)
        pl.imshow(G0, interpolation='nearest')
        pl.title('OT matrix G0')

    if fig_4:
        pl.figure(4)
        ot.plot.plot2D_samples_mat(P, Q, G0, c=[.5, .5, 1])
        pl.plot(P[:, 0], P[:, 1], '+b', label='Source samples')
        pl.plot(Q[:, 0], Q[:, 1], 'xr', label='Target samples')
        pl.legend(loc=0)
        pl.title('OT matrix with samples')

    return G0

def get_unique(row_ind_list):
    """
    Returns a list with m/gcd(P,Q) (or n/gcd(P,Q)) many sublists of unique elements
    """
    element_lists = {}
    # Iterate through the original list
    for item in row_ind_list:
        # Check if the element is already a key in the dictionary
        if item in element_lists:
            # If it's already a key, append the item to the existing list
            element_lists[item].append(item)
        else:
            # If it's not a key, create a new list with the element as the key
            element_lists[item] = [item]

    # Convert the dictionary values (lists) into a list of lists
    result_lists = list(element_lists.values())
    return result_lists

def sim_OT(m,n,it):
    """
    Function that will perform our simulations
    """
    possible_m = list(range((int(m / gcd(n,m)))+1))
    possible_n = list(range((int(n / gcd(n,m)))+1))
    m_array = np.zeros((it,len(possible_m)))
    n_array = np.zeros((it,len(possible_n)))
    for i in range(200):
        P = np.random.rand(n, 2)
        Q = np.random.rand(m, 2)

        G0 = OT_solver(P,Q,m,n,fig_1=False,fig_3=False,fig_4=False)
        G0 = np.where(G0 < 1e-10, 0, G0)
        row_ind_list = [np.count_nonzero(G0[arr]) for arr in range(len(G0))]
        col_ind_list = [np.count_nonzero((G0.T)[arr]) for arr in range(len((G0.T)))]
        unique_row, unique_col = get_unique(row_ind_list), get_unique(col_ind_list)
        
        for row in unique_row:
            row_1 = row[0]
            m_array[i,row_1] = len(row)/n

        for col in unique_col:
            col_1 = col[0]
            n_array[i,col_1] = len(col)/m

    return m_array, n_array, possible_m, possible_n

def plot_func(avg_m_statistics,avg_n_statistics,possible_m,possible_n):
    """
    Will collect data and provide our plots
    """
    ins = np.where(avg_m_statistics > 1e-10)
    plt.bar(np.array(possible_m)[ins],avg_m_statistics[ins])
    plt.xticks(np.array(possible_m)[ins])
    plt.title("Distribution of source balls: " + str(len(avg_m_statistics)-1) + " matched to target balls")
    plt.ylabel("Average frequency")
    plt.xlabel("Outgoing edges from each source ball")
    plt.show()

    ins = np.where(avg_n_statistics > 1e-10)
    plt.bar(np.array(possible_n)[ins],avg_n_statistics[ins])
    plt.xticks(np.array(possible_n)[ins])
    plt.title("Distribution of target balls: "+ str(len(avg_n_statistics)-1) +" receiving mass from source balls")
    plt.ylabel("Average frequency")
    plt.xlabel("Incoming edges from each target ball")
    plt.show()