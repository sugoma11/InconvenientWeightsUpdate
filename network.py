import torch


class Network:

    def __init__(self, reg_func, loss_func, reg_lambda: float, lr: float):

        self.layers = []
        self.reg_func = reg_func
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.loss_func = loss_func
        self.loss = None

        self.last_added_fan_out = None

    def change_lr(self, lr):
        self.lr = lr

    def add_layer(self, layer):

        print(f'Adding layer: {layer.__class__.__name__}')

        if layer.__class__.__name__ in ['Conv2DWrapper', 'MyFlatten']:
            pass

        if not self.layers or layer.__class__.__name__ not in ['Linear', 'BatchNorm1d']:

            self.layers.append(layer)

            if hasattr(layer, 'weight'):
                self.last_added_fan_out = layer.weight.shape[1]

        elif self.last_added_fan_out == layer.weight.shape[0]:

            if layer.__class__.__name__ == 'BatchNorm1d':
                self.last_added_fan_out = layer.weight.shape[0]
            else:
                self.last_added_fan_out = layer.weight.shape[1]

            self.layers.append(layer)

        else:
            self.layers.append(layer)
            # raise Exception("Wrong size of the layer!")

    def get_loss(self):
        return self.loss.detach().cpu().numpy()

    def predict(self, x):

        for layer in self.layers:
            x = layer(x, is_training=False)
            # print(f'{layer.__class__.__name__}: {x.shape}, {x.mean()}')

        return x

    def eval(self, x, target):

        for layer in self.layers:
            x = layer(x, is_training=False)

        return x, self.loss_func(x, target, is_training=False).detach().cpu().numpy()

    def my_fit(self, x: torch.Tensor, target, use_old):

        for layer in self.layers:
            x = layer(x)

        self.loss = self.loss_func(x, target).detach().cpu().numpy()

        dloss = self.loss_func.local_grad

        layers_reversed = self.layers[::-1]

        for i, layer in enumerate(layers_reversed):
            dloss = layer.backward(dloss, self.lr, use_old, self.reg_lambda, self.reg_func)

        return self.loss