import numpy as np
from data import *
from preprocessing import *
from config import *


def forward(pi, A, B, x):
    '''
    Compute alpha_t(z) defined as:
    alpha_b,t(z) = P{x_0, x_1, ..., x_t, Z_t=z | pi, A, B}
    recursively
    given the batch b of observations x = [x1, ..., xB]
    and model parameters pi, A and B

    :param pi: initial probabilities
    :param A: transition probabilities
    :param B: emission probabilities
    :param x (np.ndarray): batch of training data (numericized sentences)

    :return: alpha[b][t][z] for all b=0..batch_size-1, t=0..T-1 and for each z in preprocessed tags
    '''
    batch_size, T = x.shape
    K = len(pi)

    alpha = np.ones((batch_size, T, K))

    alpha[:, 0, :] = pi * B[:, x[:, 0]].T
    for t in range(1, T):
        alpha[:, t, :] = (alpha[:, t-1, :] @ A) * B[:, x[:, t]].T

    return alpha


def backward(pi, A, B, x):
    '''
    Backward algorithm compute beta[t][tag] defined as:
    beta[t][tag] = P{x_t+1, x_t+2, ..., x_T | Z_t = tag, pi, A, B}
    recursively
    given the batch b of observations x = [x1, ..., xB]
    and model parameters pi, A and B

    :param pi: initial probabilities
    :param A: transition probabilities
    :param B: emission probabilities
    :param x (np.ndarray): batch of training data (numericized sentences)
    :return: beta[b][t][tag] for all b=0..batch_size-1, t=0..T-1 and for each tag in tags
    '''
    batch_size, T = x.shape
    K = len(pi)

    beta = np.ones((batch_size, T, K))
    for t in reversed(range(T-1)):
        beta[:, t, :] = (A @ (B[:, x[:, t + 1]] * beta[:, t + 1, :].T)).T
    return beta


def Viterbi(pi, A, B, x):
    '''
    Viterbi algorithm predicts "the best" states z_b,0*, z_b,1*, ..., z_b,T-1*
    for evry observation in batch of observations x,
    given the model parameters A, B and pi.

    The best states z_0*, z_1*, ..., z_T-1*
    maximize the complete log likelihood:
    z_b,0*, ..., z_b,T-1* = argmax_{z_0, ..., z_T-1} P{X=x_b, z_0, ..., z_T-1 | pi, A, B}

    :param pi: initial probabilities
    :param A: transition probabilities
    :param B: emission probabilities
    :param x: batch of observations

    :return: the best paths
    '''
    batch_size, T = x.shape
    K = len(pi)

    # delta: best score up to time t in state i
    delta = np.zeros((batch_size, T, K))
    # psi: backpointers
    psi = np.zeros((batch_size, T, K), dtype=int)

    delta[:, 0, :] = pi * B[:, x[:, 0]].T  # shape (B, N)
    psi[:, 0, :] = 0

    for t in range(1, T):
        # Expand delta[:, t-1, :] to (B, N, 1)
        prev = delta[:, t - 1, :][:, :, None]  # shape (B, N, 1)

        # Transition: prev * A
        scores = prev * A[None, :, :]  # shape (B, N, N)

        # Max over previous states
        psi[:, t, :] = np.argmax(scores, axis=1)  # best prev state
        delta[:, t, :] = np.max(scores, axis=1) * B[:, x[:, t]].T

    # Best final states
    best_states = np.zeros((batch_size, T), dtype=int)
    for b in range(batch_size):
        best_states[b, T - 1] = np.argmax(delta[b, T - 1, :])

    # Backtrack
    for b in range(batch_size):
        for t in reversed(range(T - 2)):
            next_state = best_states[b, t + 1]
            best_states[b, t] = psi[b, t + 1, next_state]

    return best_states


def gamma(alpha, beta):
    '''
    Algorithm returns (batch_size, T, K) tensor defined as:
    gamma[b][t][k] = E[Z_b,t,k = 1 | pi, A, B]
    '''
    gamma = alpha * beta  # (M, T, N)
    gamma /= gamma.sum(axis=2, keepdims=True)  # normalize over hidden states
    return gamma


def xi(A, B, x, alpha, beta):
    '''
    Algorithm calculates
    xi[b, t, i, j] = E[Z_b,t,i = 1, Z_b,t+1,j = 1| pi, A, B]
    '''
    batch_size, T = x.shape
    K = alpha.shape[2]

    xi = np.zeros((batch_size, T - 1, K, K))

    for t in range(T - 1):
        # emission probs for next obs, shape (M, N)
        B_next = B[:, x[:, t + 1]].T  # (M, N)

        # combine beta with emissions
        temp = beta[:, t + 1, :] * B_next  # (M, N)

        # alpha_t @ A gives (M, N, N) after broadcasting
        xi[:, t, :, :] = alpha[:, t, :, None] * A[None, :, :] * temp[:, None, :]

        # normalize per sequence
        xi[:, t, :, :] /= xi[:, t, :, :].sum(axis=(1, 2), keepdims=True)

        return xi


def m_step(xi, gamma, x, vocab_size):
    '''
    Assumprton: Individual observations are independent of each other.
    '''
    batch_size, T, K = gamma.shape

    pi_cap = np.mean([gamma[b][0] for b in range(batch_size)], axis=0)  # shape (K,)

    xi_sum = np.sum([xi[b].sum(axis=0) for b in range(batch_size)], axis=0)  # (K, K)
    gamma_sum = np.sum([gamma[b][:-1].sum(axis=0) for b in range(batch_size)], axis=0)  # (K,)
    A_cap = xi_sum / gamma_sum[:, None]  # normalize row-wise, shape (K, K)

    B_cap = np.zeros((K, vocab_size))
    for v in range(vocab_size):
        # Mask where observation equals symbol v
        mask = (x == v)  # shape (batch_size, T), True where obs = v

        # Numerator: sum gamma over those times
        numer = 0
        denom = 0
        for b in range(batch_size):
            numer += gamma[b][:T][mask[b, :T]].reshape(-1, K).sum(axis=0)
            denom += gamma[b][:T].sum(axis=0)
        B_cap[:, v] = numer / denom

    return pi_cap, A_cap, B_cap


def EM(pi_init, A_init, B_init, x):
    pi_curr, A_curr, B_curr = pi_init, A_init, B_init
    for iter in range(50):
        # E step
        alpha_curr = forward(pi_curr, A_curr, B_curr, x)
        beta_curr = backward(pi_curr, A_curr, B_curr, x)
        gamma_curr = gamma(alpha_curr, beta_curr)
        xi_curr = xi(A_curr, B_curr, x, alpha_curr, beta_curr)

        # M step
        pi_curr, A_curr, B_curr = m_step(xi_curr, gamma_curr, x, vocab_size)
    return pi_curr, A_curr, B_curr

if __name__=='__main__':
    '''
    Recources:
    1. Hidden Markov Models
    https://web.stanford.edu/~jurafsky/slp3/A.pdf
    - Elements of HMMs
    - Forward-backward algorithm (for single observation)
    - Baum-Welch algorithm (for single observation)
    
    2. Training Hidden Markov Models with Multiple Observations – A Combinatorial Method
    https://scispace.com/pdf/training-hidden-markov-models-with-multiple-observations-a-46jcjwd03b.pdf
    - Baum-Welch algorithm for multiple observation
    '''
    tags_vocab = Tags(training_tags)
    vocab = Vocabulary(training_sentences)
    training_x, training_z = preprocess_data(training_sentences, training_tags, vocab, tags_vocab)

    batch_size, T = training_x.shape
    K = len(tags_vocab)
    vocab_size = len(vocab)

    seed = 2
    rng = np.random.default_rng(seed)
    pi_init = rng.random(K)
    pi_init /= pi_init.sum()
    A_init = rng.random((K, K))
    A_init /= A_init.sum(axis=1, keepdims=True)
    B_init = rng.random((K, vocab_size))
    B_init /= B_init.sum(axis=1, keepdims=True)

    pi_cap, A_cap, B_cap = EM(pi_init, A_init, B_init, training_x)
    best_states = Viterbi(pi_cap, A_cap, B_cap, training_x)

    '''
    1. conf dodaj A_init, B_init...
    2. uradi jedan EM algoritam i analiziraj rezutate
    3. napisi komentare
    4. Objavi na Git
    '''














