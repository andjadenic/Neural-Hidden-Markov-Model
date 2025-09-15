import torch
import torch.nn as nn


def normalize(x, dim):
    norm = x.sum(dim=dim, keepdim=True)
    norm[norm == 0] = 1
    x.div_(norm)
    return norm


class BaumWelch(nn.Module):
    def __init__(self, observations, pad_idx, init_log_parameters):
        super(BaumWelch, self).__init__()

        self.init_log_pi, self.init_log_A, self.init_log_B = init_log_parameters
        self.log_pi = self.init_log_pi
        self.log_A = self.init_log_A
        self.log_B = self.init_log_B
        self.log_parameters = init_log_parameters

        self.observations = observations
        self.tokens = set(observations.flatten().tolist())
        self.V = len(self.tokens)

        self.pad_id = pad_idx

        self.K = self.init_log_pi.shape[0]  # number of all possible hidden states
        self.Nb, self.T = self.observations.shape  # batch size, max sequence length

        # For message passing algorithm
        # Unnormalized values of alpha, beta, gamma and eta in log space
        self.log_alpha = torch.zeros(Nb, T, K)
        self.log_beta = torch.zeros(Nb, T, K)
        self.log_gamma = torch.zeros(Nb, T, K)
        self.log_xi = torch.zeros(Nb, T, K, K)


    def e_step(self):
        '''
        Function runs forward-backward algorithm
        calculate alpha, beta, gamma and eta
        (unnormalized in log space)
        for each observation in the given batch
        and for given model parameters (pi, A, B).
        '''
        log_pi, log_A, log_B = self.log_parameters

        # Forward-Backward algorithm

        # Calculating alpha
        log_alpha = torch.zeros(Nb, T, K)
        # Initialization
        log_alpha[:, 0, :] = log_pi + log_B[:, observations[:, 0]].T
        # Induction
        for t in range(1, T):
            p = log_alpha[:, t - 1, :].unsqueeze(1) + log_A.T.unsqueeze(0)
            log_alpha[:, t, :] = log_B[:, observations[:, t]].T + torch.logsumexp(p, dim=2)
        self.log_alpha = log_alpha

        # Calculating beta
        # Initialization
        log_beta = torch.zeros((Nb, T, K))
        # Recursion
        for t in reversed(range(T - 1)):
            p = log_A.unsqueeze(0) + log_B[:, observations[:, t + 1]].T.unsqueeze(1) + log_beta[:, t + 1, :].unsqueeze(1)
            log_beta[:, t, :] = torch.logsumexp(p, dim=2)
        self.log_beta = log_beta

        # Calculating gamma
        log_gamma = log_alpha + log_beta
        log_gamma -= torch.logsumexp(log_gamma, dim=2, keepdim=True)  # normalization
        self.log_gamma = log_gamma

        # Calculating eta
        log_xi = torch.ones((Nb, T, K, K))
        for t in range(T - 1):
            p = (
                    log_alpha[:, t, :].unsqueeze(2)
                    + log_A.unsqueeze(0)
                    + log_B[:, observations[:, t + 1]].T.unsqueeze(1)
                    + log_beta[:, t + 1, :].unsqueeze(1)
            )  # (Nb, K, K)
            log_xi[:, t, :, :] = p - torch.logsumexp(p.view(Nb, -1), dim=1)[:, None, None]
        self.log_xi = log_xi


    def m_step(self):
        '''
        M-step in Baum-Welch algorithm for batch of observation
        assuming observations are independent
        is done by averaging results acros batches.

        Recource:
        Training Hidden Markov Models with Multiple Observations – A Combinatorial Method
        https://scispace.com/pdf/training-hidden-markov-models-with-multiple-observations-a-46jcjwd03b.pdf
        '''
        # Pi update
        log_pi_cap = torch.logsumexp(self.log_gamma[:, 0, :], dim=0)  # sum over all batches
        log_pi_cap -= torch.logsumexp(log_pi_cap, dim=0)  # normalization
        self.log_pi = log_pi_cap

        # A update
        log_A_cap = torch.logsumexp(self.log_xi, dim=(0, 1))  # (K, K)
        log_A_cap -= (torch.logsumexp(self.log_gamma[:, :-1, :], dim=(0, 1))).unsqueeze(dim=1)
        log_A_cap -= torch.logsumexp(log_A_cap, dim=1, keepdim=True)  # normalization
        self.log_A = log_A_cap

        # B update
        Nb, T = self.observations.shape
        K, V = self.K, self.V

        # One-hot encode observations
        obs_mask = torch.zeros((Nb, T, V))
        obs_mask.scatter_(2, self.observations.unsqueeze(-1), 1.0)

        # Remove pad_idx influence
        obs_mask[:, :, self.pad_id] = 0

        # Expand gamma for broadcasting
        log_gamma_expanded = self.log_gamma.unsqueeze(-1)  # (Nb, T, K, 1)

        # Expected emission counts for each state k and vocab v
        log_B_num = torch.logsumexp(
            log_gamma_expanded + obs_mask.unsqueeze(2).log(),  # (Nb, T, K, V)
            dim=(0, 1)
        )  # (K, V)

        # Total expected counts per state, ignoring pads
        gamma_masked = self.log_gamma.masked_fill(
            self.observations.eq(self.pad_id).unsqueeze(-1),
            float("-inf")
        )
        log_B_den = torch.logsumexp(gamma_masked, dim=(0, 1)).unsqueeze(1)  # (K, 1)

        # Normalized log-probs
        log_B_cap = log_B_num - log_B_den

        # Force pad_idx emissions to -inf (impossible)
        log_B_cap[:, self.pad_id] = float("-inf")

        self.log_B = log_B_cap

        # Update parameters
        self.log_parameters = (self.log_pi, self.log_A, self.log_B)



if __name__ == "__main__":
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
        
        3. Explanation of log-sum-exp trick:
        https://www.youtube.com/watch?v=-RVM21Voo7Q&list=PLsmUMh77gCCpFLcWtohvl5oNuQknAkqYi&index=12
        '''

    # Example
    Nb = 2  # batch size
    K = 3  # number of all possible hidden states
    T = 4  # max number of time steps
    V = 5  # vocabulary size

    init_log_pi = torch.tensor([-10, -20, -300], dtype=torch.float)
    init_log_A = torch.tensor([[-10, -10, -10],
                                    [-200, -100, -155.5],
                                    [-230, -10000, -55]], dtype=torch.float)
    init_log_B = torch.tensor([[-10, -10, -10, -10, -10, ],
                                    [-210, -310, -410, -510, -610],
                                    [-210, -210, -210, -110, -110]], dtype=torch.float)
    init_log_parameters = (init_log_pi, init_log_A, init_log_B)

    observations = torch.tensor([[2, 2, 0, 0],
                                      [3, 2, 1, 1]], dtype=torch.long)

    bw_model = BaumWelch(observations=observations,
                         pad_idx=0,
                         init_log_parameters=init_log_parameters)