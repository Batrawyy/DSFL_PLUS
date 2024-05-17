import os
from collections import defaultdict
from logging import Logger
from typing import DefaultDict, List, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from algorithm.base import BaseSerialClientTrainer, BaseServerHandler
from dataset import PartitionedDataset
from utils import CustomDataset,LoadedDataset

class DSFLSerialClientTrainer(BaseSerialClientTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        num_clients: int,
        state_dict_dir: str,
        logger: Logger,
        cuda=False,
        device=None,
        personal=False,
    ) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self.id_to_state_dict_path: DefaultDict[int, str] = defaultdict(str)
        self.state_dict_dir = state_dict_dir
        os.makedirs(self.state_dict_dir, exist_ok=True)
        self.logger = logger
        print(f"Initialized DSFLSerialClientTrainer for {num_clients} clients.")

    def setup_dataset(self, dataset: PartitionedDataset):
        self.dataset = dataset
        self.public_dataset = dataset.get_dataset(type="public")
        print("Dataset setup completed.")

    def setup_kd_optim(self, epochs: int, batch_size: int, lr: float):
        self.kd_epochs = epochs
        self.kd_batch_size = batch_size
        self.kd_lr = lr
        self.kd_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.kd_lr)
        self.kd_criterion = torch.nn.KLDivLoss(reduction="batchmean")
        print("Knowledge distillation optimizer setup completed.")

    def local_process(self, payload: list, id_list: list[int], round: int):
        global_logits = payload[0]
        global_indices = payload[1]
        next_indices = payload[2]
        self.round = round
        accuracies = []
        print(f"Round {round}: Starting local processing for {len(id_list)} clients.")
        for id in tqdm(id_list, desc=f"Round {round}: Training", leave=False):
            self.current_client_id = id
            data_loader = self.dataset.get_dataloader(
                type="private", batch_size=self.batch_size, cid=id
            )
            if self.id_to_state_dict_path[id] == "":
                self.id_to_state_dict_path[id] = os.path.join(
                    self.state_dict_dir, f"{id:03}.pt"
                )
            self.train(
                state_dict_path=self.id_to_state_dict_path[id],
                global_logits=global_logits,
                global_indices=global_indices,
                train_loader=data_loader,
            )
            accuracies.append(accuracy)
            pack = self.predict(next_indices)
            self.cache.append(pack)
        average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        print(f"Average Training Accuracy for Round {round}: {average_accuracy:.2f}%")
        return average_accuracy
    def train(
        self,
        state_dict_path: str,
        global_logits: torch.Tensor,
        global_indices,
        train_loader: DataLoader,
    ) -> None:
        if os.path.isfile(state_dict_path):
            self.model.load_state_dict(torch.load(state_dict_path)["model_state_dict"])
            self.optimizer.load_state_dict(
                torch.load(state_dict_path)["optimizer_state_dict"]
            )
            self.kd_optimizer.load_state_dict(
                torch.load(state_dict_path)["kd_optimizer_state_dict"]
            )
        else:
            self.setup_optim(self.epochs, self.batch_size, self.lr)
            self.setup_kd_optim(self.kd_epochs, self.kd_batch_size, self.kd_lr)
        self.model.train()
        correct = 0
        total = 0
        if global_logits is not None:
            public_subset = Subset(self.public_dataset, global_indices)
            public_loader = DataLoader(public_subset, batch_size=self.batch_size)
            public_logits_loader = DataLoader(
                LoadedDataset(data=torch.unbind(global_logits, dim=0), labels=None),
                batch_size=self.kd_batch_size,
            )
            for _ in range(self.kd_epochs):
                for batch_idx, ((data, _), logit) in enumerate(
                    zip(public_loader, public_logits_loader)
                ):
                    if self.cuda:
                        data = data.cuda(self.device)
                        logit = logit.cuda(self.device)
                    output = F.log_softmax(self.model(data), dim=1)
                    logit = logit.squeeze(1)
                    kd_loss = self.kd_criterion(output, logit)
                    self.kd_optimizer.zero_grad()
                    kd_loss.backward()
                    self.kd_optimizer.step()
        for _ in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "kd_optimizer_state_dict": self.kd_optimizer.state_dict(),
            },
            state_dict_path,
        )
        print(f"Training complete for client {self.current_client_id}, Accuracy: {accuracy:.2f}%")
        return accuracy
        #print(f"Training complete for client {self.current_client_id}, state saved to {state_dict_path}.")

    def predict(self, public_indices: torch.Tensor) -> List[torch.Tensor]:
        self.model.eval()
        tmp_local_logits: List[torch.Tensor] = []
        with torch.no_grad():
            predict_subset = Subset(self.public_dataset, public_indices.tolist())
            predict_loader = DataLoader(
                predict_subset, batch_size=min(self.batch_size, len(public_indices))
            )
            for data, _ in predict_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                output = self.model(data)
                logits = F.softmax(output, dim=1)
                tmp_local_logits.extend([logit.detach().cpu() for logit in logits])
        local_logits = torch.stack(tmp_local_logits)
        local_indices = torch.tensor(public_indices.tolist())
        print(f"Prediction complete for public data indices {local_indices.tolist()}.")
        return [local_logits, local_indices]

class DSFLServerHandler(BaseServerHandler):
    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        sample_ratio: float,
        cuda: bool,
        temperature: float,
        public_size_per_round: int,
        logger,
    ):
        super().__init__(model, global_round, sample_ratio, cuda)
        self.global_logits: Union[torch.Tensor, None] = None
        self.global_indices: Union[torch.Tensor, None] = None
        self.temperature = temperature
        self.public_size_per_round = public_size_per_round
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = logger
        print("DSFLServerHandler initialized.")

    def setup_kd_optim(self, kd_epochs: int, kd_batch_size: int, kd_lr: float):
        self.kd_epochs = kd_epochs
        self.kd_batch_size = kd_batch_size
        self.kd_lr = kd_lr
        self.kd_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.kd_lr)
        self.kd_criterion = torch.nn.KLDivLoss(reduction="batchmean")
        print("Server knowledge distillation optimizer setup completed.")

    def setup_dataset(self, dataset: PartitionedDataset) -> None:
        self.public_dataset = dataset.get_dataset(type="public")
        self.set_next_public_indices(size=self.public_size_per_round)
        print("Server dataset setup completed.")

    def set_next_public_indices(self, size: int) -> None:
        assert hasattr(self.public_dataset, "__len__")
        size = min(size, len(self.public_dataset))
        shuffled_indices = torch.randperm(len(self.public_dataset))
        self.global_next_indices = shuffled_indices[:size]
        print(f"Next public indices set for size {size}.")

    def global_update(self, buffer: list) -> None:
        logits_list = [ele[0] for ele in buffer]
        indices_list = [ele[1] for ele in buffer]
        global_logits_stack = defaultdict(list)
        for logits, indices in zip(logits_list, indices_list):
            for logit, indice in zip(logits, indices):
                global_logits_stack[indice.item()].append(logit)
        global_logits: List[torch.Tensor] = []
        global_indices: List[int] = []
        for indice, logits in global_logits_stack.items():
            global_indices.append(indice)
            mean_logit = torch.stack(logits).mean(dim=0).cpu()
            era_logit = F.softmax(mean_logit / self.temperature, dim=0)
            global_logits.append(era_logit)
        self.model.train()
        global_subset = Subset(self.public_dataset, global_indices)
        global_loader = DataLoader(global_subset, batch_size=self.kd_batch_size)
        print(global_logits)
        global_logits_loader = DataLoader(
            LoadedDataset(data=global_logits, labels=None),
            batch_size=self.kd_batch_size,
        )
        for _ in range(self.kd_epochs):
            for batch_idx, ((data, target), logit) in enumerate(
                zip(global_loader, global_logits_loader)
            ):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                    logit = logit.cuda(self.device)
                output = F.log_softmax(self.model(data), dim=1)
                logit = logit.squeeze(1)
                kd_loss = self.kd_criterion(output, logit)
                self.kd_optimizer.zero_grad()
                kd_loss.backward()
                self.kd_optimizer.step()
        self.global_indices = torch.tensor(global_indices)
        self.global_logits = torch.stack(global_logits)
        self.set_next_public_indices(size=self.public_size_per_round)
		    #print("Global model update completed, prepared for next round.")

    @property
    def downlink_package(self) -> List[Union[torch.Tensor, None]]:
        print("Sending downlink package to clients.")
        return [self.global_logits, self.global_indices, self.global_next_indices]
