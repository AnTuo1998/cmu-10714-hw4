import time
import numpy as np
from models import *
from needle import backend_ndarray as nd
import needle.nn as nn
import needle as ndl
import sys
sys.path.append('../python')

device = ndl.cpu()

### CIFAR-10 training ###


def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                  lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
                      clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ## BEGIN YOUR SOLUTION
    total_loss = 0.
    n_samples = 0
    acc = 0

    if opt:
        model.train()
    else:
        model.eval()

    h = None
    for i in range(0, len(data)-1, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len,
                                  device=device, dtype=dtype)
        y_pred, h = model(X, h)
        loss = loss_fn(y_pred, y)

        # detach h
        if isinstance(h, tuple):
            h = tuple([h_.detach() for h_ in h])
        else:
            h = h.detach()

        if opt:
            opt.reset_grad()
            loss.backward()
            if clip:
                pass
            opt.step()

        # last batch could be less
        # so must use total_loss over all samples
        # instead of over all batch
        total_loss += loss.numpy() * y.shape[0]
        n_samples += y.shape[0]
        acc += np.equal(y_pred.numpy().argmax(axis=1), y.numpy()).sum()

    return acc / n_samples, total_loss / n_samples
    ## END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
              lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
              device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = loss_fn()
    for ep in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len,
                                              loss_func, opt, clip,
                                              device, dtype)

    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
                 device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len,
                                          loss_fn(), None, None,
                                          device, dtype)

    return avg_acc, avg_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(
        corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary),
                          hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
