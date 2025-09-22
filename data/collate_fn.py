from pythonProject1.data.preprocessing import *
from torch.nn.utils.rnn import pad_sequence


def collate_fn(samples_list):
    '''
    samples_list is a list of tuples (x, y)
    '''
    xs, ys = zip(*samples_list)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0)
    return (xs_padded, ys_padded)