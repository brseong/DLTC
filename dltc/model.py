from dataclasses import dataclass
from enum import Enum
from turtle import st
import torch
from torch import nn
from torch.nn import functional as F
from . import layer as slayer

@dataclass
class ModelInfo:
    layer_shape:list
    weight_lr:float
    l2_lr:float
    scale:float
    in_channels:int|None = None
    flatten_size:int|None = None

def VE(msg):
    raise ValueError(msg)

# %%
class SimpleFFN(nn.Module):
    def __init__(self, minfo:ModelInfo):
        super(SimpleFFN, self).__init__()
        self.scale = minfo.scale
        self.shape = minfo.layer_shape
        self.weight_lr, self.l2_lr = minfo.weight_lr, minfo.l2_lr
        self.layers: list[slayer.SNNLayer] = nn.Sequential() # type: ignore
        assert len(minfo.layer_shape) != 0, f"Wrong layer shape: {minfo.layer_shape}"
        for f_in, f_out in zip(minfo.layer_shape[:-1], minfo.layer_shape[1:]):
            self.layers.append(slayer.SNNLayer(in_size=f_in, out_size=f_out))
        # self.layer_in = layer.SNNLayer(in_size=784, out_size=1000)
        # self.layer_out = layer.SNNLayer(in_size=1000, out_size=self.num_classes)

    def forward(self, image, label):
        image = self.scale * (-image + 1)
        image = torch.exp(image.view(image.size(0), -1))  # Flatten the image

        x_out = image
        for layer in self.layers:
            x_out = layer(x_out)
        # layerin_out = self.layer_in.forward(image)
        # layerout_out = self.layer_out.forward(layerin_out)

        output_real = F.one_hot(label, num_classes=self.shape[-1]).float()
        layerout_groundtruth = torch.cat(
            [x_out, output_real], dim=1
        )  # (Batch, Class)
        loss = torch.mean(
            torch.stack([slayer.loss_func(x.unsqueeze(0)) for x in layerout_groundtruth])
        )

        wsc = sum(layer.w_sum_cost() for layer in self.layers)
        # wsc = self.layer_in.w_sum_cost() + self.layer_out.w_sum_cost()
        l2c = sum(layer.l2_cost() for layer in self.layers)
        # l2c = self.layer_in.l2_cost() + self.layer_out.l2_cost()

        cost = loss + self.weight_lr * wsc + self.l2_lr * l2c
        correct = (torch.argmax(-x_out, dim=1) == label).float().mean()

        return cost, correct
    
class SimpleCNN(nn.Module):
    def __init__(self, minfo:ModelInfo):
        super(SimpleCNN, self).__init__()
        self.scale = minfo.scale
        self.shape = minfo.layer_shape
        self.weight_lr, self.l2_lr = minfo.weight_lr, minfo.l2_lr
        self.layers: list[slayer.SCNNLayer|slayer.SNNLayer|nn.MaxPool2d] = nn.Sequential() # type: ignore
        
        assert len(minfo.layer_shape) != 0, f"Wrong layer shape: {minfo.layer_shape}"
        assert "Flatten" in minfo.layer_shape, f"layer_shape must include 'Flatten' layer."
        assert minfo.in_channels != None, f"In channels must not be None."
        
        mlp_shape = list[int]()
        for i, item in enumerate(minfo.layer_shape):
            if item == "Flatten":
                mlp_shape = minfo.layer_shape[i+1:]
                break
            elif not isinstance(item, tuple):
                raise ValueError(f"Wrong layer input: {item}.")
            elif item[0] == "P":
                self.layers.append(nn.MaxPool2d(item[1]))
                continue
            else:
                kernel_size, num_channels, stride = item
                self.layers.append(
                    slayer.SCNNLayer(kernel_size, minfo.in_channels, num_channels, stride)
                )
        
        f_in = minfo.flatten_size if minfo.flatten_size is not None else VE("Flatten size not given.")
        for f_out in mlp_shape:
            self.layers.append(slayer.SNNLayer(in_size=f_in, out_size=f_out))
            f_in = f_out

    def forward(self, image, label):
        image = self.scale * (-image + 1)
        image = torch.exp(image.view(image.size(0), 1, 28, 28))

        x_out = image
        for layer in self.layers:
            x_out = layer(x_out)

        output_real = F.one_hot(label, num_classes=self.shape[-1]).float()
        layerout_groundtruth = torch.cat(
            [x_out, output_real], dim=1
        )  # (Batch, Class)
        loss = torch.mean(
            torch.stack([slayer.loss_func(x.unsqueeze(0)) for x in layerout_groundtruth])
        )

        wsc = sum(layer.w_sum_cost() for layer in self.layers)
        l2c = sum(layer.l2_cost() for layer in self.layers)

        cost = loss + self.weight_lr * wsc + self.l2_lr * l2c
        correct = (torch.argmax(-x_out, dim=1) == label).float().mean()

        return cost, correct