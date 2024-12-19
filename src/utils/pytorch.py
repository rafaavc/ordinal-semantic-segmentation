import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter
from time import time as get_time

writer = None


def finalize_pytorch():
    writer.flush()
    writer.close()


def setup_pytorch(ARGS, model_folder: str):
    global writer
    use_cuda = ARGS.use_cuda and torch.cuda.is_available()
    print(f"CUDA: {use_cuda}")
    torch.manual_seed(ARGS.seed)
    ARGS.device = torch.device("cuda" if use_cuda else "cpu")
    if model_folder:
        writer = SummaryWriter(model_folder)


def train_batch(t, train_loss, args, optimizer, model, criterion, current_batch, total_batches, data, target):
    data = data.to(args.device)  # move data to device (GPU) if necessary
    if target is not None:
        target = target.to(args.device)

    optimizer.zero_grad()  # reset optimizer

    output = model(data)   # forward pass: calculate output of network for input

    loss = criterion(output, target)  # calculate loss
    loss.backward()  # backward pass: calculate gradients using automatic diff. and backprop.

    optimizer.step()  # udpate parameters of network using our optimizer


    # Logging and stats
    train_loss += loss.item()
    cur_time = get_time()
    if cur_time - t > args.log_interval or current_batch == total_batches-1:  
        print('[{}/{} batches ({:.0f}%)]\ttrain loss: {:.6f}, took {:.2f}s'.format(
                    current_batch + 1, total_batches,
                    100. * (current_batch + 1) / total_batches, train_loss / (current_batch+1), cur_time - t))
        t = cur_time
    
    return train_loss, t


def train_epoch_dnn(current_fold, current_epoch, model, train_loader, unsupervised_train_loader, optimizer, criterion, args):
    """
    Training loop for one epoch of NN training.
    :param model: The model to be trained
    :param train_loader: Data provider
    :param optimizer: Optimizer (Gradient descent update algorithm)
    :param args: NN parameters for training and inference
    :return:
    """
    model.train()  # set model to training mode (activate dropout layers for example)
    train_loss = 0
    t = get_time() # we measure the needed time
    total_batches = len(train_loader) + (0 if unsupervised_train_loader is None else len(unsupervised_train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):  # iterate over training data
        train_loss, t = train_batch(t, train_loss, args, optimizer, model, criterion, batch_idx, total_batches, data, target)

    if unsupervised_train_loader:
        print("Going through unsupervised dataset")
        for batch_idx_unsupervised, data in enumerate(unsupervised_train_loader):
            train_loss, t = train_batch(t, train_loss, args, optimizer, model, criterion, batch_idx_unsupervised + batch_idx + 1, \
                        total_batches, data, None)

    train_loss /= total_batches
    
    writer.add_scalar(f'Fold{current_fold}/loss/train', train_loss, current_epoch)
    print("Average train loss = {:.4f}".format(train_loss))


def test_dnn(current_fold, current_epoch, model, test_loader, criterion, args):
    """
    Function wich iterates over test data (eval or test set) without performing updates and calculates loss.
    :param model: The model to be tested
    :param test_loader: Data provider
    :param args: NN parameters for training and inference
    :return: cumulative test loss
    """
    model.eval()  # set model to inference mode (deactivate dropout layers for example)
    test_loss = 0  # init overall loss
    with torch.no_grad():  # do not calculate gradients since we do not want to do updates
        for data, target in test_loader:  # iterate over test data
            data, target = data.to(args.device), target.to(args.device)  # move data to device 
            output = model(data) # forward pass
            # calculate loss and add it to our cumulative loss
            test_loss += criterion(output, target).item()
    test_loss /= len(test_loader)  # calc mean loss

    tb_category = f'Fold{current_fold}/loss/val' if current_fold != None else 'Loss/test'
    writer.add_scalar(tb_category, test_loss, current_epoch)
    print('Average eval loss = {:.4f}'.format(test_loss, len(test_loader.dataset)))
    return test_loss


def inference_dnn(model, data, args):
    """
    Function calculating the actual output of the network, given some input.
    :param model: The network to be used
    :param data: Data for which the output should be calculated
    :param args: NN parameters for training and inference
    :return: output of network
    """
    model.eval()
    data = data.to(args.device)
    with torch.no_grad():
        return model(data)
