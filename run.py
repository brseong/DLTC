import os
from dltc.scripts.fmnist import train_scnn as fmnist_scnn, train_snn as fmnist_snn
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fmnist")
    parser.add_argument("--conv", action="store_true")
    args = parser.parse_args()
    
    match getattr(args, "dataset"):
        case "fmnist":
            if getattr(args, "conv"):
                fmnist_scnn.train()
            else:
                fmnist_snn.train()
        case "cifar10":
            pass
        case _: raise ValueError("Wrong dataset.")