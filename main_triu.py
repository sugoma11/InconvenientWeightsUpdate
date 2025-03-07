from sklearn.model_selection import train_test_split
import numpy as np
from copy import deepcopy
import pickle
import pandas as pd
from tqdm import tqdm
import torch
import random

from network import Network
from layers import Linear, LReLU, BatchNorm1d, CrossEntropy

def none_reg(x):
    return x

def l2_reg(x):
    return 2 * x

def torch_reg(x):
    return x

def get_batches(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]


def get_batches_rnd(x, y, batch_size):
    indices = list(range(len(x)))
    random.shuffle(indices)

    for i in range(0, len(x), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield x[batch_indices], y[batch_indices]


data = pd.read_csv("/home/sc.uni-leipzig.de/ms53dumu/InconvenientWeightsUpdate/arch/fashion-mnist_train.csv")

X = torch.from_numpy(data[data.columns[1:]].values).float().cuda()
Y = torch.from_numpy(data[data.columns[0]].values).float().cuda()

test_data = pd.read_csv("/home/sc.uni-leipzig.de/ms53dumu/InconvenientWeightsUpdate/arch/fashion-mnist_test.csv")

test_x = torch.from_numpy(test_data[test_data.columns[1:]].values).float().cuda()
test_y = torch.from_numpy(test_data[test_data.columns[0]].values).float().cuda()

test_x = test_x / 255

X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)
X_train_small = X_train_small / 255
X_test_small = X_test_small / 255

ce = CrossEntropy(optimize_exponents=True)

nn2_old_bn = Network(reg_func=torch_reg, loss_func=ce, reg_lambda=0.001, lr=0.01)

nn2_old_bn.add_layer(Linear((784, 256)))
nn2_old_bn.add_layer(BatchNorm1d(256))
nn2_old_bn.add_layer(LReLU(0.001))

nn2_old_bn.add_layer(Linear((256, 10)))

nn2_new_bn = deepcopy(nn2_old_bn)


nn3_old_bn = Network(reg_func=torch_reg, loss_func=ce, reg_lambda=0.001, lr=0.01)

nn3_old_bn.add_layer(Linear((784, 512)))
nn3_old_bn.add_layer(BatchNorm1d(512))
nn3_old_bn.add_layer(LReLU(0.001))

nn3_old_bn.add_layer(Linear((512, 256)))
nn3_old_bn.add_layer(BatchNorm1d(256))
nn3_old_bn.add_layer(LReLU(0.001))

nn3_old_bn.add_layer(Linear((256, 10)))

nn3_new_bn = deepcopy(nn3_old_bn)


nn4_old_bn = Network(reg_func=torch_reg, loss_func=ce, reg_lambda=0.001, lr=0.01)

nn4_old_bn.add_layer(Linear((784, 512)))
nn4_old_bn.add_layer(BatchNorm1d(512))
nn4_old_bn.add_layer(LReLU(0.001))

nn4_old_bn.add_layer(Linear((512, 256)))
nn4_old_bn.add_layer(BatchNorm1d(256))
nn4_old_bn.add_layer(LReLU(0.001))

nn4_old_bn.add_layer(Linear((256, 128)))
nn4_old_bn.add_layer(BatchNorm1d(128))
nn4_old_bn.add_layer(LReLU(0.001))

nn4_old_bn.add_layer(Linear((128, 10)))

nn4_new_bn = deepcopy(nn4_old_bn)

nn5_old_bn = Network(reg_func=torch_reg, loss_func=ce, reg_lambda=0.001, lr=0.01)

nn5_old_bn.add_layer(Linear((784, 512)))
nn5_old_bn.add_layer(BatchNorm1d(512))
nn5_old_bn.add_layer(LReLU(0.001))

nn5_old_bn.add_layer(Linear((512, 256)))
nn5_old_bn.add_layer(BatchNorm1d(256))
nn5_old_bn.add_layer(LReLU(0.001))

nn5_old_bn.add_layer(Linear((256, 128)))
nn5_old_bn.add_layer(BatchNorm1d(128))
nn5_old_bn.add_layer(LReLU(0.001))

nn5_old_bn.add_layer(Linear((128, 64)))
nn5_old_bn.add_layer(BatchNorm1d(64))
nn5_old_bn.add_layer(LReLU(0.001))

nn5_old_bn.add_layer(Linear((64, 10)))

nn5_new_bn = deepcopy(nn5_old_bn)


nn6_old_bn = Network(reg_func=torch_reg, loss_func=ce, reg_lambda=0.001, lr=0.01)

nn6_old_bn.add_layer(Linear((784, 512)))
nn6_old_bn.add_layer(BatchNorm1d(512))
nn6_old_bn.add_layer(LReLU(0.001))

nn6_old_bn.add_layer(Linear((512, 256)))
nn6_old_bn.add_layer(BatchNorm1d(256))
nn6_old_bn.add_layer(LReLU(0.001))

nn6_old_bn.add_layer(Linear((256, 128)))
nn6_old_bn.add_layer(BatchNorm1d(128))
nn6_old_bn.add_layer(LReLU(0.001))

nn6_old_bn.add_layer(Linear((128, 64)))
nn6_old_bn.add_layer(BatchNorm1d(64))
nn6_old_bn.add_layer(LReLU(0.001))

nn6_old_bn.add_layer(Linear((64, 32)))
nn6_old_bn.add_layer(BatchNorm1d(32))
nn6_old_bn.add_layer(LReLU(0.001))

nn6_old_bn.add_layer(Linear((32, 10)))

nn6_new_bn = deepcopy(nn6_old_bn)


nn7_old_bn = Network(reg_func=torch_reg, loss_func=ce, reg_lambda=0.001, lr=0.01)

nn7_old_bn.add_layer(Linear((784, 512)))
nn7_old_bn.add_layer(BatchNorm1d(512))
nn7_old_bn.add_layer(LReLU(0.001))

nn7_old_bn.add_layer(Linear((512, 256)))
nn7_old_bn.add_layer(BatchNorm1d(256))
nn7_old_bn.add_layer(LReLU(0.001))

nn7_old_bn.add_layer(Linear((256, 128)))
nn7_old_bn.add_layer(BatchNorm1d(128))
nn7_old_bn.add_layer(LReLU(0.001))

nn7_old_bn.add_layer(Linear((128, 64)))
nn7_old_bn.add_layer(BatchNorm1d(64))
nn7_old_bn.add_layer(LReLU(0.001))

nn7_old_bn.add_layer(Linear((64, 32)))
nn7_old_bn.add_layer(BatchNorm1d(32))
nn7_old_bn.add_layer(LReLU(0.001))

nn7_old_bn.add_layer(Linear((32, 16)))
nn7_old_bn.add_layer(BatchNorm1d(16))
nn7_old_bn.add_layer(LReLU(0.001))

nn7_old_bn.add_layer(Linear((16, 10)))

nn7_new_bn = deepcopy(nn7_old_bn)


def train_two_models(model_one, model_two, epoch, batch_size, lr_schedule=None, use_old_p=0.0, device='cpu'):
    loss_tr_old = []
    loss_val_old = []
    loss_tr_new = []
    loss_val_new = []

    acc_tr_old = []
    acc_val_old = []
    acc_tr_new = []
    acc_val_new = []

    for ep in range(epoch):

        loss_tr_old_ = []
        loss_val_old_ = []
        loss_tr_new_ = []
        loss_val_new_ = []

        acc_tr_old_ = []
        acc_val_old_ = []
        acc_tr_new_ = []
        acc_val_new_ = []

        if (lr_schedule is not None) and ep in lr_schedule:
            model_one.change_lr(lr_schedule[ep])
            model_two.change_lr(lr_schedule[ep])

        for i, batch in tqdm(enumerate(get_batches(X_train_small, y_train_small, batch_size))):
            x_batch, y_batch = map(lambda x: x.to(device), batch)

            loss_tr_old_.append(model_one.my_fit(x_batch, y_batch, use_old=True).item())

            loss_tr_new_.append(
                model_two.my_fit(x_batch, y_batch, use_old=bool(np.random.binomial(size=1, n=1, p=use_old_p))).item())  # bool(np.random.binomial(1, 0.5, size=1))))

            old_pred = model_one.predict(x_batch).argmax(dim=1)
            new_pred = model_two.predict(x_batch).argmax(dim=1)

            acc_tr_old_.append((old_pred == y_batch).sum().item() / len(old_pred))
            acc_tr_new_.append((new_pred == y_batch).sum().item() / len(new_pred))

            del x_batch
            del y_batch

            # gc.collect()
            # torch.cuda.empty_cache()

        loss_tr_old.append(sum(loss_tr_old_) / len(loss_tr_old_))
        loss_tr_new.append(sum(loss_tr_new_) / len(loss_tr_new_))

        acc_tr_old.append((sum(acc_tr_old_) / len(acc_tr_old_)))
        acc_tr_new.append((sum(acc_tr_new_) / len(acc_tr_new_)))

        for i, batch in enumerate(get_batches(X_test_small, y_test_small, batch_size)):
            x_batch, y_batch = map(lambda x: x.to(device), batch)

            old_pred, loss_old = model_one.eval(x_batch, y_batch)
            new_pred, loss_new = model_two.eval(x_batch, y_batch)

            old_pred = old_pred.argmax(dim=1)
            new_pred = new_pred.argmax(dim=1)

            loss_val_old_.append(loss_old.item())
            loss_val_new_.append(loss_new.item())

            acc_val_old_.append((old_pred == y_batch).sum().item() / len(old_pred))
            acc_val_new_.append((new_pred == y_batch).sum().item() / len(new_pred))

            del x_batch
            del y_batch

            # gc.collect()
            # torch.cuda.empty_cache()

        loss_val_old.append(sum(loss_val_old_) / len(loss_val_old_))
        loss_val_new.append(sum(loss_val_new_) / len(loss_val_new_))

        acc_val_old.append((sum(acc_val_old_) / len(acc_val_old_)))
        acc_val_new.append((sum(acc_val_new_) / len(acc_val_new_)))

        print(
            f'epoch: {ep} loss_tr_old: {loss_tr_old[-1]}, loss_tr_new: {loss_tr_new[-1]}, acc_tr_old: {acc_tr_old[-1]}, acc_tr_new: {acc_tr_new[-1]}')
        print(
            f'         loss_val_old: {loss_val_old[-1]}, loss_val_new: {loss_val_new[-1]}, acc_val_old: {acc_val_old[-1]}, acc_val_new: {acc_val_new[-1]}')
        print("-" * 100)

    num_of_layers = sum(1 for layer in model_one.layers if isinstance(layer, Linear))
    is_bn = bool(sum(1 for layer in model_one.layers if isinstance(layer, BatchNorm1d)))

    output = open(f'/home/sc.uni-leipzig.de/ms53dumu/InconvenientWeightsUpdate/NEW_DIR/bs-{batch_size};layers-{num_of_layers};-BN={is_bn}.pkl', 'wb')
    pickle.dump(
        [loss_tr_old, loss_tr_new, acc_tr_old, acc_tr_new, loss_val_old, loss_val_new, acc_val_old, acc_val_new],
        output)
    output.close()


def train_batches(model_one, model_two, epoch, init_lr_list: list, batch_sizes_list: list):
    for lr, bs in zip(init_lr_list, batch_sizes_list):
        model_one_ = deepcopy(model_one)
        model_two_ = deepcopy(model_two)

        model_one_.change_lr(lr)
        model_two_.change_lr(lr)

        lr_schedule = {10: lr * 0.05, 30: lr * 0.001}

        train_two_models(model_one_, model_two_, epoch, bs, lr_schedule, use_old_p=0.0, device='cuda:0')

        del model_one_
        del model_two_

epoch_num = 20
init_lr = 0.00075
batch_sizes = [2, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512]
lrs_list = [0.001 for i in range(1, len(batch_sizes))]

train_batches(nn2_old_bn, nn2_new_bn,  epoch=epoch_num, init_lr_list=lrs_list, batch_sizes_list=batch_sizes)
del nn2_old_bn, nn2_new_bn
train_batches(nn3_old_bn, nn3_new_bn,  epoch=epoch_num, init_lr_list=lrs_list, batch_sizes_list=batch_sizes)
del nn3_old_bn, nn3_new_bn
train_batches(nn4_old_bn, nn4_new_bn,  epoch=epoch_num, init_lr_list=lrs_list, batch_sizes_list=batch_sizes)
del nn4_old_bn, nn4_new_bn
train_batches(nn5_old_bn, nn5_new_bn,  epoch=epoch_num, init_lr_list=lrs_list, batch_sizes_list=batch_sizes)
del nn5_old_bn, nn5_new_bn
train_batches(nn6_old_bn, nn6_new_bn,  epoch=epoch_num, init_lr_list=lrs_list, batch_sizes_list=batch_sizes)
del nn6_old_bn, nn6_new_bn
train_batches(nn7_old_bn, nn7_new_bn,  epoch=epoch_num, init_lr_list=lrs_list, batch_sizes_list=batch_sizes)
del nn7_old_bn, nn7_new_bn
