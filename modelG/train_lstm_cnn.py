from __future__ import print_function
import os
import time
import torch
import argparse
import torch.nn.parallel
import torch.utils.data
import torch.backends.cudnn as cudnn
import lstm_cnn_data as lstm_cnn_dataset
import numpy as np
import deepdish

from torch.autograd import Variable
from lstm_cnn import LSTMCNN

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', default=4000, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--patch_size', default=64, type=int)
parser.add_argument('--dir', default='Photos', type=str)
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')

parser.add_argument('--learning_rate', default=0.0002, type=float)

parser.add_argument('--gpu_id', default='0', type=str, help='the id of a gpu')
parser.add_argument('--workers', default=8, type=int, help='the number of workers to load the data')
parser.add_argument('--checkpoint_folder', default='checkpoints', type=str, help='checkpoint folder')
parser.add_argument('--resume', default='', type=str, help='which checkpoint file to resume the training')

parser.add_argument('--nf', type=int, default=64, help='max size of the activation maps')
parser.add_argument('--nz', type=int, default=512, help='size of the maximum activation maps')


def create_dataset(dir, mode):
    return lstm_cnn_dataset.LSTMCNNDataset(dir, mode)


def main():
    global args
    args = parser.parse_args()

    # initialize CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if torch.cuda.is_available():
        torch.randn(8).cuda()

    # create data loader
    train_dataset = create_dataset(args.dir, 'train')
    val_dataset = create_dataset(args.dir, 'val')

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_dataset.weight, len(train_dataset.weight))
    val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_dataset.weight, len(val_dataset.weight))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             sampler=val_sampler,
                                             num_workers=args.workers)

    # create model
    net = LSTMCNN(args.image_size, 3, args.nf, args.nz, 4, args.learning_rate, args.batch_size)

    if torch.cuda.is_available():
        net.cuda()

    # create a checkpoint folder if not exists
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    # optionally resume from a checkpoint
    start_epoch = 0
    train_loss = []
    train_f1_benign = []
    train_f1_lumen = []
    train_f1_stroma = []
    train_f1_tumour = []

    val_loss = []
    val_f1_benign = []
    val_f1_lumen = []
    val_f1_stroma = []
    val_f1_tumour = []

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']

            train_loss = checkpoint['train_loss']
            train_f1_benign = checkpoint['train_f1_benign']
            train_f1_lumen = checkpoint['train_f1_lumen']
            train_f1_stroma = checkpoint['train_f1_stroma']
            train_f1_tumour = checkpoint['train_f1_tumour']

            val_loss = checkpoint['val_loss']
            val_f1_benign = checkpoint['val_f1_benign']
            val_f1_lumen = checkpoint['val_f1_lumen']
            val_f1_stroma = checkpoint['val_f1_stroma']
            val_f1_tumour = checkpoint['val_f1_tumour']

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):
        t_loss, \
        t_f1_lumen, \
        t_f1_stroma, \
        t_f1_benign, \
        t_f1_tumour, \
        t_time = train(train_loader, net)

        v_loss, \
        v_f1_lumen, \
        v_f1_stroma, \
        v_f1_benign, \
        v_f1_tumour, \
        v_time = validate(val_loader, net)

        train_loss.append(t_loss)
        train_f1_benign.append(t_f1_benign)
        train_f1_lumen.append(t_f1_lumen)
        train_f1_stroma.append(t_f1_stroma)
        train_f1_tumour.append(t_f1_tumour)

        val_loss.append(v_loss)
        val_f1_benign.append(v_f1_benign)
        val_f1_lumen.append(v_f1_lumen)
        val_f1_stroma.append(v_f1_stroma)
        val_f1_tumour.append(v_f1_tumour)

        train_print_str = 'epoch: %d ' \
                          'train_loss: %.3f ' \
                          'train_f1_lumen: %.3f ' \
                          'train_f1_stroma: %.3f ' \
                          'train_f1_benign: %.3f ' \
                          'train_f1_tumour: %.3f ' \
                          'train_time: %.3f '

        val_print_str = 'epoch: %d ' \
                        'val_loss: %.3f ' \
                        'val_f1_lumen: %.3f ' \
                        'val_f1_stroma: %.3f ' \
                        'val_f1_benign: %.3f ' \
                        'val_f1_tumour: %.3f ' \
                        'val_time: %.3f'

        print(train_print_str % (epoch, t_loss, t_f1_lumen, t_f1_stroma, t_f1_benign, t_f1_tumour, t_time))
        print(val_print_str % (epoch, v_loss, v_f1_lumen, v_f1_stroma, v_f1_benign, v_f1_tumour, v_time))

        # save checkpoint
        save_name = os.path.join(args.checkpoint_folder,
                                 'checkpoint_' + str(epoch) + '.pth')
        save_variable_name = os.path.join(args.checkpoint_folder,
                                          'variables.mat')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'train_loss': train_loss,
            'train_f1_benign': train_f1_benign,
            'train_f1_lumen': train_f1_lumen,
            'train_f1_stroma': train_f1_stroma,
            'train_f1_tumour': train_f1_tumour,
            'val_loss': val_loss,
            'val_f1_benign': val_f1_benign,
            'val_f1_lumen': val_f1_lumen,
            'val_f1_stroma': val_f1_stroma,
            'val_f1_tumour': val_f1_tumour},
            {'train_loss': np.array(train_loss),
             'train_f1_benign': np.array(train_f1_benign),
             'train_f1_lumen': np.array(train_f1_lumen),
             'train_f1_stroma': np.array(train_f1_stroma),
             'train_f1_tumour': np.array(train_f1_tumour),
             'val_loss': np.array(val_loss),
             'val_f1_benign': np.array(val_f1_benign),
             'val_f1_lumen': np.array(val_f1_lumen),
             'val_f1_stroma': np.array(val_f1_stroma),
             'val_f1_tumour': np.array(val_f1_tumour)},
            filename=save_name,
            variable_name=save_variable_name)


def train(loader, net):
    loss_meter = AverageMeter()
    acc_meter = [AverageMeter() for _ in range(4)]
    time_meter = AverageMeter()

    loss_meter.reset()
    [acc_meter[i].reset() for i in range(4)]
    time_meter.reset()

    # switch to train mode
    net.train()

    t = time.time()
    count = 0
    for input, label in loader:
        sub_time = time.time()
        input_var = [Variable(x.cuda(non_blocking=True), requires_grad=False) for x in input]   # changed
        label_var = Variable(label.cuda(non_blocking=True), requires_grad=False)                # changed

        # set the grad to zero
        net.optimizer.zero_grad()

        # run the model
        predict = net(input_var)

        # calculate loss
        loss = net.loss(predict, label_var)

        # backward and optimizer
        loss.backward()
        net.optimizer.step()

        # record loss
        loss_meter.update(loss.data[0], input[0].size(0))

        # measure and record f1
        acc = net.f1score(predict, label_var)
        [acc_meter[i].update(acc[i], input[0].size(0)) for i in range(4)]

        print_str = 'count: %d ' \
                    'batch_loss: %.3f ' \
                    'batch_f1_lumen: %.3f ' \
                    'batch_f1_stroma: %.3f ' \
                    'batch_f1_benign: %.3f ' \
                    'batch_f1_tumour: %.3f ' \
                    'time: %.3f'

        print(print_str % (
        count, loss_meter.avg, acc_meter[0].avg, acc_meter[1].avg, acc_meter[2].avg, acc_meter[3].avg,
        time.time() - sub_time))

        # update count
        count += 1

        # measure elapsed time
        time_meter.update(time.time() - t)

    return loss_meter.avg, acc_meter[0].avg, acc_meter[1].avg, acc_meter[2].avg, acc_meter[3].avg, time_meter.sum


def validate(loader, net):
    loss_meter = AverageMeter()
    acc_meter = [AverageMeter() for _ in range(4)]
    time_meter = AverageMeter()

    loss_meter.reset()
    [acc_meter[i].reset() for i in range(4)]
    time_meter.reset()

    # switch to evaluation mode
    net.eval()

    t = time.time()
    count = 0
    for input, label in loader:
        sub_time = time.time()
        input_var = [Variable(x.cuda(async=True), requires_grad=False) for x in input]
        label_var = Variable(label.cuda(async=True), requires_grad=False)

        # run the model
        predict = net(input_var)

        # calculate loss
        loss = net.loss(predict, label_var)

        # record loss
        loss_meter.update(loss.data[0], input[0].size(0))

        # measure and record f1
        acc = net.f1score(predict, label_var)
        [acc_meter[i].update(acc[i], input[0].size(0)) for i in range(4)]

        print_str = 'count: %d ' \
                    'batch_loss: %.3f ' \
                    'batch_f1_lumen: %.3f ' \
                    'batch_f1_stroma: %.3f ' \
                    'batch_f1_benign: %.3f ' \
                    'batch_f1_tumour: %.3f ' \
                    'time: %.3f'

        print(print_str % (
        count, loss_meter.avg, acc_meter[0].avg, acc_meter[1].avg, acc_meter[2].avg, acc_meter[3].avg,
        time.time() - sub_time))

        # update count
        count += 1

        # measure elapsed time
        time_meter.update(time.time() - t)

    return loss_meter.avg, acc_meter[0].avg, acc_meter[1].avg, acc_meter[2].avg, acc_meter[3].avg, time_meter.sum


def save_checkpoint(state,
                    variables,
                    filename='checkpoint.pth.tar', variable_name='variables.h5'):
    torch.save(state, filename)
    deepdish.io.save(variable_name, variables)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
