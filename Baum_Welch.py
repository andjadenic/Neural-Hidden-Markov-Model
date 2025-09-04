import numpy as np
from data import *
from preprocessing import *


def forward(pi, A, B, q):
    '''
    Compute alpha_t(id(tag)) defined as:
    alpha_t(tag) = P{x_0, x_1, ..., x_t, Z_t=tag | pi, A, B}
    recursively
    given the query (observations, sentence) q = [x_0, x_1, ..., x_T-1]
    and parameters pi, A and B

    :param pi: initial probabilities
    :param A: transition probabilities
    :param B: emission probabilities
    :param q: query (observations)

    :return: alpha[t][id(tag)] for all t=0..T-1 and for each tag in tags
    '''
    T = len(q)
    K = len(pi)
    alpha = np.ones((T, K))

    for t in range(T):
        if t == 0:
            alpha[t,:] = pi * B[:,q[0]]
        else:
            for k in range(K):
                alpha[t, k] = np.matmul(alpha[t - 1], A[:, k]) * B[k, q[t]]
    return alpha


def backward(pi, A, B, q):
    '''
    Backward algorithm compute beta[t][tag] defined as:
    beta[t][tag] = P{x_t+1, x_t+2, ..., x_T | Z_t = tag, pi, A, B}
    recursively
    given the query (observations) q = [x_0, x_1, ..., x_T-1]
    and parameters pi, A and B

    :param pi: initial probabilities
    :param A: transition probabilities
    :param B: emission probabilities
    :param q: query (observations)
    :return: beta[t][tag] for all t=0..T-1 and for each tag in tags
    '''
    T = len(q)
    K = len(pi)
    beta = np.ones((T, K))

    for t in reversed(range(T)):
        if t < T-1:
            for i in range(K):
                beta[t, i] = np.matmul(A[i, :] * B[:, q[t+1]], beta[t+1, :])
    return beta


def Viterbi(pi, A, B, q):
    '''
    Viterbi algorithm predicts "the best" states z_0*, z_1*, ..., z_T-1*
    given the observations (query) q = [x_0, x_1, ..., x_T-1]
    and model parameters A, B and pi.
    The best states z_0*, z_1*, ..., z_T-1*
    maximize the complete log likelihood:
    z_0*, ..., z_T-1* = argmax_{z_0, ..., z_T-1} P{X=q, z_0, ..., z_T-1 | pi, A, B}

    :param pi: initial probabilities
    :param A: transition probabilities
    :param B: emission probabilities
    :param q: query (observations)

    :return: path = [z_0*, ..., z_T-1*]
    '''
    T = len(q)
    K = len(pi)
    v = np.zeros((T, K))
    b = np.zeros((T, K))

    for t in range(T):
        if t == 0:
            v[t, :] = pi[:] * B[:, q[t]]
        else:
            for j in range(K):
                p = B[j, q[t]] * (v[t - 1, :] * A[:, j])
                v[t, j], b[t, j] = np.max(p), np.argmax(p)
    s = int(np.argmax(v[T - 1, :]))
    path = [s]
    for t in reversed(range(T)):
        if t > 0:
            s = int(b[t, s])
            path.append(s)
        else:
            break
    path.reverse()
    return np.array(path)


def e_step(A, alpha, beta):
    '''
    Algorithm calculate xi_t(id(tag_1), id(tag_2))
    and gamma_t(id(tag)) defined as:
        - xi_t(id(tag_1), id(tag_2)) = P{ Z_t = tag_1, Z_t+1 = tag_2| X = q, pi, A, B}.
          xi_t(tag_1, tag_2) is probability that tag_1 is hidden state at time step t
          and tag_2 is hidden state at time step t+1, given model parameters
        - gamma_t(id(tag)) = P{Z_t = tag | pi, A, B}
          gamma_t(id(tag)) is probability that tag is hidden state
          at time step t, given model parameters
    the observations (query) q = [x_0, x_1, ..., x_T-1].

    :param A: transition probabilities
    :param alpha: result of forward algorihm
    :param beta: result of backward algorihm

    :return: xi, gamma
    '''
    T = alpha.shape[0]
    K = A.shape[0]

    xi = np.zeros((T-1, K, K))
    gamma = np.zeros((T, K))

    for t in range(T):
        p = np.matmul(alpha[t, :], beta[t, :])
        gamma[t] = (alpha[t, :] * beta[t, :]) / p
        if t < T-1:
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = alpha[t,i] * A[i,j] * B[j, q[t+1]] * beta[t+1, j]
                    xi[t, i, j] /= p
    return gamma, xi


def m_step(xi, gamma, q, vocab_size):
    '''

    :param xi:
    :param gamma:
    :return:
    '''
    T = xi.shape[0]
    K = xi.shape[1]

    pi_cap = np.zeros((K,))
    A_cap = np.zeros((K, K))
    B_cap = np.zeros((K, vocab_size))

    pi_cap = gamma[0, :]

    A_cap = np.sum(xi, axis=0)
    A_cap /= np.sum(A_cap, axis=1, keepdims=True)

    for k in range(K):
        for i in range(vocab_size):
            B_cap[k, i] = 0
            for t in range(T):
                B_cap[k, i] += gamma[t, k] * int(q[t] == k)
    B_cap /= np.sum(B_cap, axis=1, keepdims=True)
    return pi_cap, A_cap, B_cap

if __name__=='__main__':

    # On HMMs: https://web.stanford.edu/~jurafsky/slp3/A.pdf

    tags = Tags(annotations)
    K = len(tags)  # number of possible hidden states (tags)

    alpha = forward(pi, A, B, q)
    beta = backward(pi, A, B, q)
    path = Viterbi(pi, A, B, q)

    gamma, xi = e_step(A, alpha, beta)
    pi_cap, A_cap, B_cap = m_step(xi, gamma, q, vocab_size)
    print(f'{pi_cap=}', '\n')
    print(f'{A_cap=}', '\n')

    '''
    0. Kako ovo uci iz batch-a primera?
    1. conf dodaj A_init, B_init...
    2. uradi jedan EM algoritam i analiziraj rezutate
    3. napisi komentare
    4. Objavi na Git
    '''














