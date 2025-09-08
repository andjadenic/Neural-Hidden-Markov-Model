import torch
import torch.nn as nn


def normalize(x, dim):
    norm = x.sum(dim=dim, keepdim=True)
    norm[norm == 0] = 1
    x.div_(norm)
    return norm


class BaumWelch(nn.Module):
    def __init__(self, padidx):
        super(BaumWelch, self).__init__()
        self.padidx = padidx

        # for message passing algorithm
        self.alpha = torch.Tensor()
        self.beta = torch.Tensor()
        self.gamma = torch.Tensor()
        self.eta = torch.Tensor()
        self.scale = torch.Tensor()

        # BUFFER TENSOR
        self.prob_trans = torch.Tensor()  # A
        self.prob_emiss = torch.Tensor()  # B
        self.prob_prior = torch.Tensor()  # pi
        self.buffer = torch.Tensor()

        self.debug = True

    def run(self, input, stats):
        #  Andja: forward-backward + gamma + eta
        '''

        :param input:
        :param stats: (log_emiss, log_trans, log_prior)
        :return:
        '''
        N, T = input.shape  # batch size, sequence length
        buffer = self.buffer
        prob_prior = self.prob_prior  # pi
        prob_trans = self.prob_trans  # A
        prob_emiss = self.prob_emiss  # B

        log_emiss, log_trans, log_prior = stats  # initial model parameters
        K = log_prior.shape[0]  # number of all possible hidden states

        prob_prior = log_prior.exp()
        prob_trans = log_trans.exp()
        prob_emiss = log_emiss.exp()

        # For handling padded sequences
        masked = input.ne(self.padidx)  # returns True for words
        masked_pad = input.eq(self.padidx)  # returns True for <PAD>

        # Message Passing
        alpha = torch.zeros(N, T, K)  # unnormalized forward messages
        scale = torch.zeros(N, T)  # sum of alphas across all K states
        beta = torch.ones(N, T, K)
        gamma = torch.zeros(N, T, K)

        eta = torch.zeros(N, T, K, K)

        # FORWARD MESSAGE
        # (1) compute the first alpha
        alpha[:, 0, :] = prob_prior.unsqueeze(0).expand(N, K)
        alpha[:, 0, :] *= prob_emiss[:, 0, :]

        # normalization
        scale[:, 0] = normalize(alpha[:, 0, :], dim=1).squeeze()

        # (2) compute the rest of alpha
        for t in range(1, T):
            emi_t = prob_emiss[:, t, :]
            prev_a = alpha[:, t - 1, :].unsqueeze(1)  # (N, 1, K)
            tran_t = prob_trans[:, t - 1, :, :]       # (N, K, K)
            curr_a = torch.bmm(prev_a, tran_t).squeeze(1) * emi_t
            alpha[:, t, :] = curr_a
            scale[:, t] = normalize(alpha[:, t, :], dim=1).squeeze()

        # BACKWARD MESSAGE
        beta[:, -1, :] /= scale[:, -1].unsqueeze(1).expand(N, K)
        buffer = torch.zeros(N, 1, K)  # store __ data

        eos = masked[:, :-1] != masked[:, 1:]  # end-of-sequence mask
        for t in reversed(range(T - 1)):
            eos_t = eos[:, t].unsqueeze(1).expand(N, K)
            emi_t = prob_emiss[:, t + 1, :]
            prev_b = beta[:, t + 1, :]
            buffer = prev_b * emi_t
            tran_t = prob_trans[:, t, :, :]
            curr_b = torch.bmm(buffer.unsqueeze(1), tran_t.transpose(1, 2)).squeeze(1)
            curr_b[eos_t] = 1
            curr_b /= scale[:, t].unsqueeze(1).expand(N, K)
            beta[:, t, :] = curr_b

        # compute posteriors
        for t in range(T):
            gamma[:, t, :] = alpha[:, t, :] * beta[:, t, :]
            # Debug code omitted but can be restored if needed

        normalize(gamma, dim=2)
        gamma *= masked.double().unsqueeze(2).expand(N, T, K)

        # Compute eta
        for t in range(T - 1):
            emi_t = prob_emiss[:, t + 1, :]
            bmsg = beta[:, t + 1, :] * emi_t
            amsg = alpha[:, t, :].unsqueeze(2)  # (N, K, 1)
            tran_t = prob_trans[:, t, :, :]
            eta_t = torch.bmm(amsg, bmsg.unsqueeze(1)) * tran_t
            z = eta_t.sum(1).sum(1).unsqueeze(1).unsqueeze(2).expand(N, K, K)
            eta_t = eta_t / z
            eta_t *= masked[:, t + 1].view(N, 1, 1).expand(N, K, K).double()
            eta[:, t, :, :] = eta_t

        prior = gamma[:, 0, :].sum(0)

        scale[masked_pad] = 1
        loglik = scale.clone().log().sum() / masked.sum()  # log-likelihood

        return (prior, eta, gamma), loglik  # #

    def argmax(self, input, stats):
        #  Andja: Viterbi
        # inference, we just need alpha message
        T = input.numel()
        prob_prior = self.prob_prior
        prob_trans = self.prob_trans
        prob_emiss = self.prob_emiss

        log_emiss, log_trans, log_prior = stats
        K = log_prior.numel()

        prob_prior = log_prior.exp()
        prob_trans = log_trans.exp()
        prob_emiss = log_emiss.exp()

        alpha = torch.zeros(T, K)
        psi = torch.zeros(T, K, dtype=torch.long)

        alpha[0, :] = prob_prior * prob_emiss[0, :]
        normalize(alpha[0, :], dim=0)

        for t in range(1, T):
            prev_a = alpha[t - 1, :].unsqueeze(1).repeat(1, K)
            z = prev_a * prob_trans[t - 1, :, :]
            val, idx = z.max(0)
            psi[t, :] = idx
            alpha[t, :] = val * prob_emiss[t, :]
            normalize(alpha[t, :], dim=0)

        val, idx = alpha[T - 1, :].max(0)
        path = torch.zeros(T, dtype=torch.long)
        path[-1] = idx.item()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path


if __name__ == "__main__":
    # ===== Minimal Working Example =====

    # Settings
    N = 2  # batch size
    T = 5  # sequence length
    K = 3  # number of hidden states
    padidx = 0

    # Create random "observations"
    # 0 = padding, >0 = token IDs
    input_seq = torch.tensor([[1, 2, 3, 0, 0],
                              [2, 3, 1, 2, 3]])

    # Create random log probabilities for HMM parameters
    torch.manual_seed(42)  # reproducibility
    log_prior = torch.log(torch.rand(K))  # (K,) unnormalized log probabilities
    log_prior -= torch.logsumexp(log_prior, dim=0)  # normalize

    log_trans = torch.log(torch.rand(N, T - 1, K, K))  # (N, T-1, K, K)
    log_trans -= torch.logsumexp(log_trans, dim=3, keepdim=True)  # https://www.youtube.com/watch?v=-RVM21Voo7Q&list=PLsmUMh77gCCpFLcWtohvl5oNuQknAkqYi&index=12

    log_emiss = torch.log(torch.rand(N, T, K))  # (N, T, K)
    log_emiss -= torch.logsumexp(log_emiss, dim=2, keepdim=True)

    stats = (log_emiss, log_trans, log_prior)

    # Instantiate model
    model = BaumWelch(padidx)

    # Run forward-backward (Baum-Welch E-step)
    (posteriors, eta, gamma), loglik = model.run(input_seq, stats)

    # Run argmax (Viterbi decoding) on the second sequence (no padding)
    seq2 = input_seq[1]  # length = 5
    path = model.argmax(seq2, stats)
    print("Most likely hidden state path:", path)