# -*- coding:utf-8 -*-
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import os.path
#from chainer import cuda
from chainer import Variable
from models.DNC import DNC
from models.DNC import onehot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--x', '-x', default=2, type=int, help='input dim')
    parser.add_argument('--y', '-y', default=2, type=int, help='output dim')
    # input.data.shape = (1 , x * y )
    parser.add_argument('--n', '-n', default=2, type=int, help='memory slot num')
    parser.add_argument('--w', '-w', default=2, type=int, help='dim of memory slot')
    parser.add_argument('--r', '-r', default=1, type=int, help='number of read heads')
    parser.add_argument('--epoch', default=1000, type=int, help='number of read heads')

    args = parser.parse_args()
    print(args)

    model = DNC(args.x,args.y,args.n,args.w,args.r)
    #optimizer = chainer.optimizers.Adam()
    optimizer = chainer.optimizers.SMORMS3()
    optimizer.setup(model)
    model.reset_state()

    xp = chainer.cuda.cupy if args.gpu >= 0 else np

    # TODO fix generate dictionary (like used in EmbbedID)
    data = xp.array([[0,0],[0,1],[1,0],[1,1]], np.float32)
    teacher = [ np.array([elem[0] or elem[1]] , dtype=np.int32) for elem in data  ]


    for epoch in range(args.epoch):
        loss = 0
        acc = 0
        for datum,t in zip(data , teacher):
            model.reset_state()
            for seq in datum:
                # reshape (1 , len of variable dim)
                # TODO fix input with dictionay
                x = onehot(seq , 2).reshape(1,2)
                y = model(x)
            #t = F.expand_dims(Variable(t) , axis = 0)
            temp = F.softmax_cross_entropy(y , t)
            loss += F.softmax_cross_entropy(y , t)

        model.cleargrads()
        #loss.grad = np.ones(loss.data.shape, dtype=np.float32)
        loss.backward()
        optimizer.update()
        loss.unchain_backward()


    for datum,t in zip(data , teacher):
        model.reset_state()
        for seq in datum:
            x = onehot(seq , 2).reshape(1,2)
            y = model(x)
        print datum , np.argmax(y.data[0])


    model.reset_state()
    x = onehot(1 , 2).reshape(1,2)
    _ = model(x)
    x = onehot(1 , 2).reshape(1,2)
    _ = model(x)
    x = onehot(0 , 2).reshape(1,2)
    y = model(x)

    print "1 or 1 or 0 : " , np.argmax(y.data[0])
