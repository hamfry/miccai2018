import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, torch.nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, np.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


class LSTMCNN(nn.Module):
    def __init__(self, image_size, nc, nf, nz, n_class, lr, batch_size):
        super(LSTMCNN, self).__init__()

        n = np.sqrt(image_size)
        n = int(n)

        self.batch_size = batch_size
        self.hidden_dim = nz
        self.vgg = nn.Sequential()
        # 64x64
        self.vgg.add_module('input-conv', nn.Conv2d(nc, nf, 4, 2, 1, bias=False))
        self.vgg.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))
        for i in range(n - 5):
            # state size. (ngf) x 32 x 32
            self.vgg.add_module('pyramid-{0}-{1}-conv'.format(nf * 2 ** i, nf * 2 ** (i + 1)),
                                nn.Conv2d(nf * 2 ** (i), nf * 2 ** (i + 1), 4, 2, 1, bias=False))
            self.vgg.add_module('pyramid-{0}-batchnorm'.format(nf * 2 ** (i + 1)),
                                nn.BatchNorm2d(nf * 2 ** (i + 1)))
            self.vgg.add_module('pyramid-{0}-relu'.format(nf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf*8) x 4 x 4
        self.vgg.add_module('encode-conv', nn.Conv2d(nf * 2 ** (n - 5), nz, 4))

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTMCell(nz, self.hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.linear3 = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.linear4 = nn.Linear(self.hidden_dim, n_class)

        self.apply(weights_init)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(batch_size, self.hidden_dim).cuda()),
                Variable(torch.zeros(batch_size, self.hidden_dim).cuda()))

    def forward(self, img_list):
        hidden = self.init_hidden(img_list[0].size(0))

        outs = []
        for img in img_list:
            encoded = self.vgg(img)
            hidden = self.lstm(encoded.view(encoded.size(0), encoded.size(1)), hidden)
            outs.append(hidden[0])

        outs = F.dropout(self.linear3(torch.cat(outs, 1)), p=0.2)
        outs = self.linear4(outs)

        return outs

    def loss(self, predict, label):

        predict = [predict]
        loss = 0
        for pred in predict:
            loss += self.cross_entropy(pred, label)

        return loss

    def f1score(self, predict, label):

        predict = [predict]

        tp = [0] * 4
        fp = [0] * 4
        fn = [0] * 4

        for pred in predict:
            pred = pred.data.cpu().numpy()
            lab = label.data.cpu().numpy()

            pred = np.argmax(pred, axis=1)

            for i in range(4):
                tp[i] = np.sum(np.logical_and(pred == i, lab == i))
                fp[i] = np.sum(np.logical_and(pred == i, lab != i))
                fn[i] = np.sum(np.logical_and(pred != i, lab == i))

        f1 = []
        for i in range(4):
            sum_prec = tp[i] + fp[i]
            sum_rec = tp[i] + fn[i]
            if sum_prec == 0 or sum_rec == 0:
                f1.append(0)
            else:
                precision = tp[i] / float(tp[i] + fp[i])
                recall = tp[i] / float(tp[i] + fn[i])
                if precision == 0 and recall == 0:
                    f1.append(0)
                else:
                    f1.append(2 * precision * recall / (precision + recall))

        return [0 if np.isnan(x) else x for x in f1]

    def optimizer(self):
        return
