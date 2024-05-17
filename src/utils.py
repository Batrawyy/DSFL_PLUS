import os
import random
from collections import defaultdict
from typing import Optional
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def seed_everything(seed: int) -> None:
    print(f"Setting seed: {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CompatibleSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        print(f"Initialized CompatibleSubset with {len(indices)} indices")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]  # Unpack the image and label
        if self.transform:
            image = self.transform(image)
        return image, int(label)
class LoadedDataset(Dataset):
    """
    Custom Dataset for PyTorch
    Args:
        data (list): list of data
        labels (list): list of labels
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    def __init__(self, data: list, labels: Optional[list], transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.targets = self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)
        if self.labels is None:
            return sample

        label = self.labels[idx]
        return sample, label
class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, max_images=100):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.image_files = self.image_files[:max_images]  # Limit to 100 images
        self.labels = []
        for img_file in self.image_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            print(f"Reading label from: {label_path}")
            with open(label_path, 'r') as file:
                label = file.readline().strip().split()[0]  # Only take the first element
            self.labels.append(label)
        self.targets = self.labels
        print("utils in")
        print(self.targets)
        
        print(f"CustomDataset loaded with {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')  # Load and convert to RGB
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, int(label)
    def get_targets(self):
        return self.targets

def even_class_split(dataset: Dataset, size_list: list[int]) -> list[list[int]]:
    class_indices = defaultdict(list)
    assert hasattr(dataset, "targets"), "Dataset must have 'targets' attribute"
    for i, target in enumerate(dataset.targets):
        class_indices[target].append(i)

    results = []
    print(f"Performing even class split with sizes: {size_list}")

    for i, size in enumerate(size_list):
        clipped_indices = []
        per_class_size = size // len(class_indices)
        if size % len(class_indices) != 0:
            raise ValueError(f"Size {size} must be divisible by number of classes {len(class_indices)}")

        for indices in class_indices.values():
            np.random.shuffle(indices)
            clipped_indices.extend(indices[:per_class_size])

        results.append(clipped_indices)

    print("Even class split completed")
    return results

def client_inner_dirichlet_partition(targets, num_clients, num_classes, dir_alpha, client_sample_nums, verbose=True):
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    #print(targets)
    # Generate class priors using a Dirichlet distribution
    class_priors = np.random.dirichlet([dir_alpha] * num_classes, size=num_clients)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    

    # Prepare index lists for each class
    idx_list = [np.where(targets == str(i))[0] for i in range(num_classes)]
    class_amount = [len(ids) for ids in idx_list]
    print(class_amount, "-----------------" , idx_list , "----", targets )
    if verbose:
        print("Class priors and initial index distributions:")
        for i, amounts in enumerate(class_amount):
            print(f"Class {i}: {amounts} samples")

    # Initialize client indices
    client_indices = [np.zeros(num, dtype=np.int64) for num in client_sample_nums]

    print("Starting Dirichlet partitioning")
    for cid in range(num_clients):
        print(f"Allocating samples to client {cid}")
        for sample_num in range(client_sample_nums[cid]):
            found = False
            while not found:
                class_sample = np.random.uniform()
                class_idx = np.argmax(class_sample <= prior_cumsum[cid])
                if class_amount[class_idx] > 0:
                    # Assign and remove an index from the available indices of the chosen class
                    #print(idx_list[class_idx][:-1])
      
                    selected_index = idx_list[class_idx][-1]
                    idx_list[class_idx]=idx_list[class_idx][:-1]
          
                    client_indices[cid][sample_num] = selected_index
                    class_amount[class_idx] -= 1
                    found = True
                    if verbose:
                        print(f"Assigned index {selected_index} (class {class_idx}) to client {cid}")
                # else:
                #     # Optionally print a warning if no indices are available for the randomly chosen class
                #     if verbose:
                #         print(f"No available indices for class {class_idx}, retrying...")
    
    # Summarize the final distribution of indices to each client
    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    if verbose:
        print("Partitioning complete with client distributions:")
        for k, v in client_dict.items():
            print(f"Client {k} has {len(v)} samples")

    return client_dict

