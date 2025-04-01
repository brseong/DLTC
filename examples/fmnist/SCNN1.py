import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SNNLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(SNNLayer, self).__init__()
        self.MAX_SPIKE_TIME = 1e5
        self.in_size_real = in_size
        self.out_size = out_size
        self.in_size = in_size + 1
        self.weight = nn.Parameter(torch.Tensor(self.in_size, self.out_size))
        nn.init.xavier_uniform_(self.weight)
        self.in_size_real = in_size
        self.delay = nn.Parameter(
            torch.Tensor(self.in_size_real).normal_() / (self.in_size_real**0.5)
        )

    def forward(self, layer_in):
        layer_in = layer_in * torch.exp(F.relu(self.delay))
        batch_num = layer_in.shape[0]
        bias_layer_in = torch.ones(batch_num, 1, device=layer_in.device)  # device 추가
        layer_in = torch.cat([layer_in, bias_layer_in], dim=1)

        # PyTorch에서는 top_k가 (values, indices)를 반환합니다.
        _, input_sorted_indices = torch.topk(
            -layer_in, self.in_size, dim=1, sorted=False
        )
        input_sorted = torch.gather(layer_in, 1, input_sorted_indices)
        input_sorted_outsize = input_sorted.unsqueeze(2).repeat(1, 1, self.out_size)
        weight_sorted = torch.gather(
            self.weight.unsqueeze(0).repeat(batch_num, 1, 1),
            1,
            input_sorted_indices.unsqueeze(2).repeat(1, 1, self.out_size),
        )
        weight_input_mul = weight_sorted * input_sorted_outsize
        weight_sumed = torch.cumsum(weight_sorted, dim=1)
        weight_input_sumed = torch.cumsum(weight_input_mul, dim=1)
        out_spike_all = weight_input_sumed / torch.clamp(weight_sumed - 1, min=1e-10)
        out_spike_ws = torch.where(
            weight_sumed < 1,
            self.MAX_SPIKE_TIME
            * torch.ones_like(
                out_spike_all, device=out_spike_all.device
            ),  # device 추가
            out_spike_all,
        )
        out_spike_large = torch.where(
            out_spike_ws < input_sorted_outsize,
            self.MAX_SPIKE_TIME
            * torch.ones_like(out_spike_ws, device=out_spike_ws.device),  # device 추가
            out_spike_ws,
        )
        input_sorted_outsize_slice = input_sorted_outsize[:, 1:, :]
        input_sorted_outsize_left = torch.cat(
            [
                input_sorted_outsize_slice,
                self.MAX_SPIKE_TIME
                * torch.ones(
                    batch_num,
                    1,
                    self.out_size,
                    device=input_sorted_outsize_slice.device,
                ),  # device 추가
            ],
            dim=1,
        )
        out_spike_valid = torch.where(
            out_spike_large > input_sorted_outsize_left,
            self.MAX_SPIKE_TIME
            * torch.ones_like(
                out_spike_large, device=out_spike_large.device
            ),  # device 추가
            out_spike_large,
        )
        out_spike = torch.min(out_spike_valid, dim=1).values
        return out_spike

    def w_sum_cost(self):
        threshold = 1.0
        part1 = threshold - torch.sum(self.weight, dim=0)
        part2 = F.relu(part1)  # ReLU 활성화 함수 사용
        return torch.mean(part2)

    def l2_cost(self):
        w_sqr = torch.square(self.weight)
        return torch.mean(w_sqr)


class SCNNLayer(nn.Module):
    def __init__(self, kernel_size=3, in_channel=1, out_channel=1, strides=1):
        super(SCNNLayer, self).__init__()
        self.MAX_SPIKE_TIME = 1e5
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.strides = strides
        self.kernel = SNNLayer(
            in_size=self.kernel_size * self.kernel_size * self.in_channel,
            out_size=self.out_channel,
        )

    def forward(self, layer_in):
        input_size = layer_in.shape
        # input_size = tf.shape(layer_in)
        patches = F.unfold(
            layer_in,
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.kernel_size // 2,
            stride=self.strides,
        )
        # patches의 shape: (N, C * kernel_size * kernel_size, L), 여기서 L은 패치의 수
        # patches = tf.image.extract_patches(
        #     images=layer_in,
        #     sizes=[1, self.kernel_size, self.kernel_size, 1],
        #     strides=[1, self.strides, self.strides, 1],
        #     rates=[1, 1, 1, 1],
        #     padding="SAME",
        # )
        patches = patches.reshape(
            input_size[0], self.in_channel * self.kernel_size * self.kernel_size, -1
        )
        patches = patches.transpose(1, 2)  # (N, L,  C * kernel_size * kernel_size)
        # patches_flatten = tf.reshape(
        #     patches,
        #     [input_size[0], -1, self.in_channel * self.kernel_size * self.kernel_size],
        # )
        patches_infpad = torch.where(
            patches < 0.1,
            self.MAX_SPIKE_TIME
            * torch.ones_like(patches, device=patches.device),  # device 추가
            patches,
        )
        # patches_infpad = tf.compat.v1.where(
        #     tf.less(patches_flatten, 0.1),
        #     self.MAX_SPIKE_TIME * tf.ones_like(patches_flatten),
        #     patches_flatten,
        # )

        # SNNLayer에 각 패치를 통과시킴
        img_raw = torch.stack(
            [
                self.kernel(patches_infpad[:, i, :])
                for i in range(patches_infpad.shape[1])
            ],
            dim=0,
        ).transpose(0, 1)
        # img_raw = tf.map_fn(self.kernel.forward, patches_infpad)
        img_reshaped = img_raw.reshape(
            input_size[0],
            input_size[2] // self.strides,  # 수정됨
            input_size[3] // self.strides,  # 수정됨
            self.out_channel,
        )
        # img_reshaped = tf.reshape(
        #     img_raw,
        #     [
        #         input_size[0],
        #         tf.cast(tf.math.ceil(input_size[1] / self.strides), tf.int32),
        #         tf.cast(tf.math.ceil(input_size[2] / self.strides), tf.int32),
        #         self.out_channel,
        #     ],
        # )
        return img_reshaped


def loss_func(both):
    """
    function to calculate loss, refer to paper p.7, formula 11
    :param both: a tensor, it put both layer output and expected output together, its' shape
            is [batch_size,out_size*2], where the left part is layer output(real output), right part is
            the label of input(expected output), the tensor both should be looked like this:
            [[2.13,3.56,7.33,3.97,...0,0,1,0,...]
             [3.14,5.56,2.54,15.6,...0,0,0,1,...]...]
                ↑                   ↑
             layer output           label of input
    :return: a tensor, it is a scalar of loss
    """
    output = both[:, : both.shape[1] // 2]
    index = both[:, both.shape[1] // 2 :]
    z1 = torch.exp(-(output * index).sum(dim=1))
    z2 = torch.exp(-output).sum(dim=1)
    loss = -torch.log(torch.clamp(z1 / torch.clamp(z2, min=1e-10), min=1e-10, max=1))
    return loss.mean()  # 평균 반환
    # return loss


def max_pool_layer(x, size, stride, name):
    x = F.max_pool2d(x, size, stride, padding=0)  # 수정됨
    return x
