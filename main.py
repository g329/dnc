# -*- coding:utf-8 -*-
import argparse
from collections import namedtuple
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

    data_file = open("data.txt","r")
    data_lines = data_file.readlines()
    data_file.close()
    ChainerData = namedtuple("ChainerData","x t")
    data = []

    for _line in data_lines:
        _x , _t =  _line.rstrip("\n").split(",")
        x = [ elem for elem in _x ]
        t = str(_t)
        data.append(ChainerData(x=x,t=t))


    train_data = data[:100]
    test_data = data[100:]


    xp = chainer.cuda.cupy if args.gpu >= 0 else np

    # make dict input -> one-hot vector number
    # http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=1142&lang=jp
    input_dict = {}
    input_dict.update( { i : elem for i , elem in enumerate( "a b c d e f g h i j k l m n o p q r s t u v w x y z".split() ) } )
    input_dict.update( { elem : i for i , elem in enumerate( "a b c d e f g h i j k l m n o p q r s t u v w x y z".split() ) } )
    input_length =  len(input_dict) / 2
    input_dict.update({  input_length : "<ANS>"} )

    output_dict  = {}
    output_dict.update({ i + 1 : str(elem) for i , elem in enumerate(range(2,500))} )
    output_dict.update({ str(elem) : i for i , elem in enumerate(range(2,500))})
    output_length = len(output_dict) / 2
    output_dict.update({output_length : "<END>"})


    model = DNC(input_length,output_length,args.n,args.w,args.r)
    #optimizer = chainer.optimizers.Adam()
    optimizer = chainer.optimizers.SMORMS3()
    optimizer.setup(model)
    model.reset_state()


    # tuning
    max_output_length = 10

    for epoch in range(args.epoch):
        loss = 0
        acc = 0
        for data in train_data:
            model.reset_state()
            for __seq in data.x:
                # reshape (1 , len of variable dim)
                # TODO fix input with dictionay

                seq = input_dict[__seq]
                x = onehot(seq , input_length).reshape(1,input_length)
                y = model(x)

            # 1 : batchsize
            t = np.array(data.t, dtype=np.int32 ).reshape(1)
            temp = F.softmax_cross_entropy(y , t)
            loss += F.softmax_cross_entropy(y , t)

        print epoch , ") " , loss.data
        model.cleargrads()
        loss.backward()
        optimizer.update()
        loss.unchain_backward()

    # -*- test -*-
    print("===== test =====")
    loss = 0
    for data in test_data:
        model.reset_state()
        for __seq in data.x:

            seq = input_dict[__seq]
            x = onehot(seq , input_length).reshape(1,input_length)
            y = model(x)

        # 1 : batchsize
        t = np.array(data.t, dtype=np.int32 ).reshape(1)
        loss += F.softmax_cross_entropy(y , t)

    print "test loss :" , loss.data

