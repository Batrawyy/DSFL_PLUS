import os
from typing import Optional

import pandas as pd
import torch
import torchvision
from fedlab.contrib.dataset import Subset
from fedlab.utils.dataset.functional import (
    balance_split,
    hetero_dir_partition,
    shards_partition,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils import CustomDataset, CompatibleSubset
from torchvision.transforms import Compose, ToTensor, Normalize

from utils import client_inner_dirichlet_partition, even_class_split

CLASS_NUM = {
    "cifar10": 10,
    "mnist": 10,
    "fmnist": 10,
    "cifar100": 100,
    "custom_dataset": 8,
}

class PartitionedDataset:
    def __init__(
        self,
        root: str,
        path: str,
        num_clients: int,
        partition: str,
        num_shards_per_client: int,
        dir_alpha: float,
        task: str,
        public_private_split: Optional[str],
        public_size: int,
        private_size: int,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.partition = partition
        self.num_shards_per_client = num_shards_per_client
        self.dir_alpha = dir_alpha
        self.task = task
        self.public_private_split = public_private_split
        self.public_size = public_size
        self.private_size = private_size
        self.num_classes = CLASS_NUM[self.task]
        
        print(f"Initializing dataset preprocessing for task: {self.task}")

        if self.task in ["cifar10", "cifar100"]:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.test_transform = self.transform
        elif self.task in ["mnist", "fmnist"]:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
            self.test_transform = transforms.ToTensor()

        self.preprocess()

    def preprocess(self):
        """Preprocess dataset and save to local file."""
        print(f"Preprocessing the dataset stored at {self.path}")

        if not os.path.exists(self.path):
            os.makedirs(self.root, exist_ok=True)
            os.makedirs(self.path, exist_ok=True)
            os.mkdir(os.path.join(self.path, "private"))
            os.mkdir(os.path.join(self.path, "public"))
            print(f"Created directories at {self.path}")

        match self.task:
            case "custom_dataset":
                transform = Compose([
                    ToTensor(),
                    #Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                images_dir = '/content/drive/MyDrive/12_GP24_Mohamed_Energy-Efficient Video/Dataset/My Dataset/train/images'
                labels_dir = '/content/drive/MyDrive/12_GP24_Mohamed_Energy-Efficient Video/Dataset/My Dataset/train/labels'
                self.trainset = CustomDataset(images_dir=images_dir, labels_dir=labels_dir,transform=transform)
                print('here')
                print(self.trainset.targets)
                images_dir = '/content/drive/MyDrive/12_GP24_Mohamed_Energy-Efficient Video/Dataset/My Dataset/test/images'
                labels_dir = '/content/drive/MyDrive/12_GP24_Mohamed_Energy-Efficient Video/Dataset/My Dataset/test/labels'
                self.testset = CustomDataset(images_dir=images_dir, labels_dir=labels_dir, transform=transform)
                #print(f"Custom dataset loaded with {len(self.trainset)} training samples and {len(self.testset)} test samples")
            case "cifar10":
                self.trainset = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True)
                self.testset = torchvision.datasets.CIFAR10(root=self.root, train=False, download=True, transform=self.test_transform)
                print(f"CIFAR10 loaded with {len(self.trainset)} training samples and {len(self.testset)} test samples")
            case "cifar100":
                self.trainset = torchvision.datasets.CIFAR100(root=self.root, train=True, download=True)
                self.testset = torchvision.datasets.CIFAR100(root=self.root, train=False, download=True, transform=self.test_transform)
                print(f"CIFAR100 loaded with {len(self.trainset)} training samples and {len(self.testset)} test samples")
            case "imagenet":
                self.trainset = torchvision.datasets.ImageNet(root=self.root, split="train", download=True)
                self.testset = torchvision.datasets.ImageNet(root=self.root, split="val", download=True, transform=self.test_transform)
                print(f"ImageNet loaded with {len(self.trainset)} training samples and {len(self.testset)} test samples")
            case "mnist":
                self.trainset = torchvision.datasets.MNIST(root=self.root, train=True, download=True)
                self.testset = torchvision.datasets.MNIST(root=self.root, train=False, download=True, transform=self.test_transform)
                print(f"MNIST loaded with {len(self.trainset)} training samples and {len(self.testset)} test samples")
            case "fmnist":
                self.trainset = torchvision.datasets.FashionMNIST(root=self.root, train=True, download=True)
                self.testset = torchvision.datasets.FashionMNIST(root=self.root, train=False, download=True, transform=self.test_transform)
                print(f"FashionMNIST loaded with {len(self.trainset)} training samples and {len(self.testset)} test samples")
            case _:
                raise ValueError(f"Invalid dataset task: {self.task}")

        trainset_targets = self.trainset.targets
        print(self.testset.targets)

        if self.public_private_split is not None:
            print(f"Total training samples available: {len(self.trainset)}")
            print(f"Requested public + private sizes: {self.public_size + self.private_size}")

            assert self.public_size + self.private_size <= len(self.trainset)
            if self.public_private_split == "even_class":
                public_indices, private_indices = even_class_split(dataset=self.trainset, size_list=[self.public_size, self.private_size])
                self.private_indices = private_indices
                print(f"Public/private split done with {len(public_indices)} public indices and {len(private_indices)} private indices")
            elif self.public_private_split == "random_sample":
                total_indices = torch.randperm(len(self.trainset))
                public_indices = total_indices[:self.public_size]
                private_indices = total_indices[self.public_size:self.public_size + self.private_size]
                subset_index_to_original_index = {idx: i for idx, i in enumerate(private_indices)}
                print("indexxxxx")
                print(total_indices,'-----------',subset_index_to_original_index)
                print(f"Random sample split done with {len(public_indices)} public indices and {len(private_indices)} private indices")
            else:
                raise ValueError(f"Invalid public_private_split: {self.public_private_split}")

            trainset_targets = [trainset_targets[i] for i in private_indices]
            subset_index_to_original_index = {idx: original_index for idx, original_index in enumerate(private_indices)}

        # Client partitioning logic
        match self.partition:
            case "shards":
                client_dict = shards_partition(targets=trainset_targets, num_clients=self.num_clients, num_shards=self.num_clients * self.num_shards_per_client)
                print(f"Shard partitioning done for {self.num_clients} clients")
            case "hetero_dir":
                client_dict = hetero_dir_partition(targets=trainset_targets, num_clients=self.num_clients, num_classes=CLASS_NUM[self.task], dir_alpha=self.dir_alpha)
                print(f"Heterogeneous Dirichlet partitioning done for {self.num_clients} clients")
            case "client_inner_dirichlet":
                print(trainset_targets, self.num_clients, CLASS_NUM[self.task], self.dir_alpha, balance_split(self.num_clients, len(trainset_targets)))
                client_dict = client_inner_dirichlet_partition(targets=trainset_targets, num_clients=self.num_clients, num_classes=CLASS_NUM[self.task], dir_alpha=self.dir_alpha, client_sample_nums=balance_split(self.num_clients, len(trainset_targets)), verbose=True)
                print(f"Inner Dirichlet partitioning done for {self.num_clients} clients")
            case _:
                raise ValueError(f"Invalid partition method: {self.partition}")

        # Creating subsets for each client
        subsets = dict()
        self.client_dict = dict()
        for cid in range(self.num_clients):
            if self.public_private_split is not None:
                original_indices = []

                for i in client_dict[cid]:
                    if i in subset_index_to_original_index:
                        original_indices.append(subset_index_to_original_index[i])
                    else:
                        print(f"Warning: Index {i} not found in subset_index_to_original_index")

                self.client_dict[cid] = original_indices
            else:
                self.client_dict = client_dict

            if self.task in ["covid19"]:
                subset = Subset(dataset=self.trainset, indices=self.client_dict[cid], transform=self.transform)
            else:
                subset = CompatibleSubset(dataset=self.trainset, indices=public_indices)  # Assuming this should apply the same public indices for all clients for simplicity; adjust if needed

            subsets[cid] = subset
            print(f"Created subset for client {cid} with {len(subsets[cid])} samples")

        # Saving subsets to disk
        for cid, subset in subsets.items():
            filepath = os.path.join(self.path, "private", f"{cid:03}.pkl")
            torch.save(subset, filepath)
            print(f"Saved private subset for client {cid} at {filepath}")

        # Save public subset to disk
        if self.public_private_split is not None:
            public_filepath = os.path.join(self.path, "public", "public.pkl")
            public_subset = CompatibleSubset(dataset=self.trainset, indices=public_indices)
            torch.save(public_subset, public_filepath)
            print(f"Saved public subset at {public_filepath}")

    def get_client_stats(self) -> pd.DataFrame:
        """Get statistics of the dataset for each client."""
        self.stats_dict = dict()
        for cid, indices in self.client_dict.items():
            class_count = [0] * CLASS_NUM[self.task]
            for index in indices:
                class_count[int(self.trainset.targets[index])] += 1
            self.stats_dict[cid] = class_count

        stats_df = pd.DataFrame.from_dict(
            self.stats_dict,
            orient="index",
            columns=list(map(str, range(CLASS_NUM[self.task]))),
        )
        print(f"Generated statistics for clients")
        return stats_df

    def get_dataset(self, type, cid=None) -> Dataset:
        """Load dataset for client with client ID ``cid`` from local file.

        Args:
            type (str): Dataset type, can be ``"private"``, ``"public"`` or ``"test"``.
            cid (int, optional): client id
        """
        match type:
            case "private":
                assert cid is not None
                dataset = torch.load(
                    os.path.join(self.path, type, f"{cid:03}.pkl".format(cid))
                )
            case "public":
                dataset = torch.load(os.path.join(self.path, type, f"{type}.pkl"))
            case "test":
                dataset = self.testset
            case _:
                raise ValueError(f"Invalid dataset type: {type}")
        return dataset

    def get_dataloader(self, type: str, batch_size: int, cid=None) -> DataLoader:
        """Generate a DataLoader for the specified dataset type and client ID."""
        print(f"Generating DataLoader for type '{type}' and client ID {cid}")

        dataset = self.get_dataset(type, cid)
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty for type '{type}' and client ID {cid}")

        return DataLoader(dataset, batch_size=batch_size, shuffle=(type != "test"))

