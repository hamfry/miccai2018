from __future__ import print_function
import os
import time
import torch
import argparse
import torch.nn.parallel
import torch.utils.data
import torch.backends.cudnn as cudnn
import lstm_cnn_data_test as lstm_cnn_dataset
import numpy as np

from torch.autograd import Variable
from lstm_cnn import LSTMCNN
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', default=4000, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--patch_size', default=64, type=int)
parser.add_argument('--dir', default='Photos_fat_cat_go_down_and_eat_bird', type=str)
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')

parser.add_argument('--learning_rate', default=0.0002, type=float)

parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--workers', default=12, type=int)
parser.add_argument('--checkpoint_folder', default='checkpoints', type=str)
parser.add_argument('--resume', default='checkpoints_same_lstm_777/checkpoint_9.pth', type=str)

parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
parser.add_argument('--nf', type=int, default=64)


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
    test_dataset = create_dataset(args.dir, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
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
        v_loss, \
        v_f1_lumen, \
        v_f1_stroma, \
        v_f1_benign, \
        v_f1_tumour, \
        v_time = validate(test_loader, net)

        val_loss.append(v_loss)
        val_f1_benign.append(v_f1_benign)
        val_f1_lumen.append(v_f1_lumen)
        val_f1_stroma.append(v_f1_stroma)
        val_f1_tumour.append(v_f1_tumour)

        val_print_str = 'epoch: %d ' \
                          'val_loss: %.3f ' \
                          'val_f1_lumen: %.3f ' \
                          'val_f1_stroma: %.3f ' \
                          'val_f1_benign: %.3f ' \
                          'val_f1_tumour: %.3f ' \
                          'val_time: %.3f'

        print(val_print_str % (epoch, v_loss, v_f1_lumen, v_f1_stroma, v_f1_benign, v_f1_tumour, v_time))


def validate(loader, net):
    loss_meter = AverageMeter()
    acc_meter = [AverageMeter() for _ in range(4)]
    time_meter = AverageMeter()

    loss_meter.reset()
    [acc_meter[i].reset() for i in range(4)]
    time_meter.reset()

    # switch to train mode
    net.eval()

    t = time.time()
    count = 0

    all_pred = []
    all_lab = []

    for input, label in loader:

        sub_time = time.time()
        input_var = [Variable(x.cuda(async=True), requires_grad=False) for x in input]
        label_var = Variable(label.cuda(async=True), requires_grad=False)

        # run the model
        predict = net(input_var)
        # predict = predict[0] + predict[1] + predict[2] + predict[3]

        # measure and record f1
        pred = predict.data.cpu().numpy()
        lab = label_var.data.cpu().numpy()

        pred = np.argmax(pred, axis=1)

        all_pred.append(pred)
        all_lab.append(lab)

    all_pred = np.hstack(all_pred)
    all_lab = np.hstack(all_lab)

    tp = [0] * 4
    fp = [0] * 4
    fn = [0] * 4
    for i in range(4):
        tp[i] = np.sum(np.logical_and(all_pred == i, all_lab == i))
        fp[i] = np.sum(np.logical_and(all_pred == i, all_lab != i))
        fn[i] = np.sum(np.logical_and(all_pred != i, all_lab == i))

    f1 = [0] * 4
    for i in range(4):
        sum_prec = tp[i] + fp[i]
        sum_rec = tp[i] + fn[i]
        if sum_prec == 0 or sum_rec == 0:
            f1[i] = 0
        else:
            precision = tp[i] / float(tp[i] + fp[i])
            recall = tp[i] / float(tp[i] + fn[i])
            if precision == 0 and recall == 0:
                f1[i] = 0
            else:
                f1[i] = 2 * precision * recall / (precision + recall)

    confusion = confusion_matrix(all_lab, all_pred)
    confusion = confusion / (np.sum(confusion, axis=1).astype(np.float).reshape(-1, 1))
    confusion = np.round(confusion, 3)

    return loss_meter.avg, acc_meter[0].avg, acc_meter[1].avg, acc_meter[2].avg, acc_meter[3].avg, time_meter.sum

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
