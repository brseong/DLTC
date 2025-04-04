
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

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


# %%
def get_data(train_or_test, BATCH_SIZE=128):
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