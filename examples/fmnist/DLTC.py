#!/usr/bin/env python
import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# from tensorpack import tfv1 as tf
# from tensorpack import *
# from tensorpack.dataflow import dataset, BatchData, imgaug
# from tensorpack.tfutils.summary import *
# from tensorpack.utils.gpu import get_num_gpu
# from tensorpack.dataflow import dataset
from tensorboardX import SummaryWriter  # 텐서보드 지원용

sys.path.append("..")
import SCNN1

K = 100
K2 = 1e-2
TRAINING_BATCH = 10
scale = 2


class FashionMNISTDataset(Dataset):
    def __init__(self, train, transform=None):
        self.data = datasets.FashionMNIST(
            root="./data", train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label


class Model(nn.Module):
    # class Model(ModelDesc):
    def __init__(self, cifar_classnum):
        super(Model, self).__init__()
        self.num_classes = cifar_classnum
        self.layer_in = SCNN1.SNNLayer(in_size=784, out_size=1000)
        self.layer_out = SCNN1.SNNLayer(in_size=1000, out_size=10)

    def forward(self, image, label):
        image = scale * (-image + 1)
        image = torch.exp(image.view(image.size(0), -1))  # Flatten the image

        layerin_out = self.layer_in.forward(image)
        layerout_out = self.layer_out.forward(layerin_out)

        output_real = F.one_hot(label, num_classes=self.num_classes).float()
        layerout_groundtruth = torch.cat([layerout_out, output_real], dim=1)
        loss = torch.mean(
            torch.stack([SCNN1.loss_func(x) for x in layerout_groundtruth])
        )

        wsc = self.layer_in.w_sum_cost() + self.layer_out.w_sum_cost()
        l2c = self.layer_in.l2_cost() + self.layer_out.l2_cost()

        cost = loss + K * wsc + K2 * l2c
        correct = (torch.argmax(-layerout_out, dim=1) == label).float().mean()

        return cost, correct

    # def inputs(self):
    #     return [
    #         tf.TensorSpec((None, 28, 28), tf.float32, "input"),
    #         tf.TensorSpec((None,), tf.int32, "label"),
    #     ]

    # def build_graph(self, image, label):
    #     image = scale * (-image + 1)
    #     print("input shape", image.shape)
    #     image = tf.reshape(tf.exp(image), [TRAINING_BATCH, 784])

    #     layer_in = SCNN1.SNNLayer(in_size=784, out_size=1000, n_layer=1, name="layer1")
    #     layer_out = SCNN1.SNNLayer(in_size=1000, out_size=10, n_layer=2, name="layer2")

    #     layerin_out = layer_in.forward(image)
    #     layerout_out = layer_out.forward(layerin_out)

    #     output_real = tf.one_hot(label, 10, dtype=tf.float32)
    #     layerout_groundtruth = tf.concat([layerout_out, output_real], 1)
    #     loss = tf.reduce_mean(
    #         tf.map_fn(SCNN1.loss_func, layerout_groundtruth), name="cost"
    #     )

    #     wsc = layer_in.w_sum_cost() + layer_out.w_sum_cost()
    #     l2c = layer_in.l2_cost() + layer_out.l2_cost()

    #     K = 100
    #     K2 = 1e-2
    #     cost = loss + K * wsc + K2 * l2c
    #     tf.summary.scalar("cost", cost)
    #     correct = tf.cast(
    #         tf.nn.in_top_k(predictions=-layerout_out, targets=label, k=1),
    #         tf.float32,
    #         name="correct",
    #     )

    #     # monitor training error
    #     add_moving_summary(tf.reduce_mean((correct), name="accuracy"))

    #     return cost

    # def optimizer(self):
    #     lr = tf.compat.v1.train.exponential_decay(
    #         learning_rate=1e-2,
    #         global_step=get_global_step_var(),
    #         decay_steps=int(50000 / TRAINING_BATCH),
    #         decay_rate=(1e-4 / 1e-2) ** (1.0 / 70),
    #         staircase=True,
    #         name="learning_rate",
    #     )
    #     tf.summary.scalar("lr", lr)
    #     return tf.compat.v1.train.AdamOptimizer(lr)


def get_data(train_or_test, cifar_classnum=10, BATCH_SIZE=128):
    is_train = train_or_test == "train"

    # ds = dataset.FashionMnist(train_or_test)
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    dataset = FashionMNISTDataset(train=is_train, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=is_train)
    return dataloader

    # if is_train:
    #     augmentors = [
    #         imgaug.CenterPaste((32, 32)),
    #         imgaug.RandomCrop((28, 28)),
    #     ]
    # else:
    #     augmentors = [
    #         # imgaug.CenterCrop((28, 28)),
    #     ]
    # ds = AugmentImageComponent(ds, augmentors)
    # ds = BatchData(ds, BATCH_SIZE)
    # return ds


# def get_config(cifar_classnum, BATCH_SIZE):
#     # prepare dataset
#     dataset_train = get_data("train", cifar_classnum, BATCH_SIZE)
#     dataset_test = get_data("test", cifar_classnum, BATCH_SIZE)

#     nr_tower = max(get_num_gpu(), 1)
#     batch = BATCH_SIZE
#     total_batch = batch * nr_tower
#     print("total batch", total_batch)

#     input = QueueInput(dataset_train)
#     input = StagingInput(input, nr_stage=1)

#     return TrainConfig(
#         model=Model(cifar_classnum),
#         data=input,
#         callbacks=[
#             ModelSaver(),  # save the model after every epoch
#             GPUUtilizationTracker(),
#             EstimatedTimeLeft(),
#             InferenceRunner(  # run inference(for validation) after every epoch
#                 dataset_test,  # the DataFlow instance used for validation
#                 [ScalarStats("cost"), ClassificationError("correct")],
#             ),
#             MaxSaver("validation__correct"),
#         ],
#         # ],
#         max_epoch=70,
#     )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    train_loader = get_data("train", args.classnum, args.batch)
    test_loader = get_data("test", args.classnum, args.batch)

    # Model
    model = Model(args.classnum).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(1e-4 / 1e-2) ** (1.0 / 70)
    )

    # TensorBoard writer
    writer = SummaryWriter(
        logdir=os.path.join("train_log", "delay" + str(args.classnum))
    )

    # Training loop
    for epoch in range(70):  # max_epoch=70
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            cost, correct = model(images, labels)
            cost.backward()
            optimizer.step()

            total_loss += cost.item() * images.size(0)
            total_correct += correct.item() * images.size(0)
            total_samples += images.size(0)

            # Log training info
            writer.add_scalar(
                "lr",
                optimizer.param_groups[0]["lr"],
                epoch * len(train_loader) + batch_idx,
            )
            writer.add_scalar(
                "train_loss", cost.item(), epoch * len(train_loader) + batch_idx
            )
            writer.add_scalar(
                "train_accuracy", correct.item(), epoch * len(train_loader) + batch_idx
            )

        scheduler.step()
        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct / total_samples
        print(
            f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}"
        )

        # Validation loop
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                cost, correct = model(images, labels)

                val_loss += cost.item() * images.size(0)
                val_correct += correct.item() * images.size(0)
                val_samples += images.size(0)

        val_loss = val_loss / val_samples
        val_accuracy = val_correct / val_samples
        print(
            f"Epoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        # Log validation info
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_accuracy", val_accuracy, epoch)

        # Save model checkpoint (optional)
        torch.save(
            model.state_dict(),
            os.path.join("train_log", f"model_epoch_{epoch + 1}.pth"),
        )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="comma separated list of GPU(s) to use.")
    # parser.add_argument('--load', help='load model')
    parser.add_argument("--classnum", help="10 for fmnist", type=int, default=10)
    parser.add_argument(
        "--batch", type=int, default=TRAINING_BATCH, help="batch per GPU"
    )
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train(args)
    # with tf.Graph().as_default():
    #     logger.set_logger_dir(os.path.join("train_log", "delay" + str(args.classnum)))
    #     config = get_config(args.classnum, args.batch)

    #     num_gpu = get_num_gpu()
    #     print("total gpus", num_gpu)
    #     trainer = (
    #         SimpleTrainer()
    #         if num_gpu <= 1
    #         else SyncMultiGPUTrainerParameterServer(num_gpu)
    #     )
    #     launch_train_with_config(config, trainer)
