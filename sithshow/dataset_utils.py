import numpy as np

def random_split(X, Y, train_proportion=.7):
    """
    given a dataset with leading batch dimension, 
    return a training and testing set
    """
    n_obs = len(X)
    
    rand_indeces = np.random.permutation(n_obs)    
    
    rand_X = X[rand_indeces]
    rand_Y = Y[rand_indeces]
    
    idx = int(train_proportion*n_obs)
    
    return rand_X[:idx], rand_Y[:idx], rand_X[idx:], rand_Y[idx:]
    

# from https://docs.kidger.site/equinox/examples/train_rnn/
def dataloader(arrays, batch_size, n_passes=None):
    """given a a tuple of arrays with leading batch dimension, repeatedly serve batches of data"""
    pass_i = 0

    if n_passes is None: 
        cond = lambda x: True    
    else:
        cond = lambda x: x < n_passes    
    
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    
    while cond(pass_i):
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
        
        pass_i += 1

# helper functions modified from https://github.com/pyro-ppl/numpyro/blob/master/numpyro/examples/datasets.py
def pad_sequence(sequences, pad_start=True):
    # like torch.nn.utils.rnn.pad_sequence with batch_first=True
    max_length = max(x.shape[0] for x in sequences)
    padded_sequences = []
    for x in sequences:
        pad = [(0, 0)] * np.ndim(x)
        
        if pad_start:
            pad[0] = (max_length - x.shape[0], 0)
        else:
            pad[0] = (0, max_length - x.shape[0])
            
        
        padded_sequences.append(np.pad(x, pad))

    return np.stack(padded_sequences)