import os
from examples.fmnist import DLTC, DLTCCNN
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fmnist")
    parser.add_argument("--conv", action="store_true")
    args = parser.parse_args()
    
    match getattr(args, "dataset"):
        case "fmnist":
            if getattr(args, "conv", False):
                DLTC.train()
            else:
                DLTCCNN.train()
        case "cifar10":
            pass
        case _: raise ValueError("Wrong dataset.")