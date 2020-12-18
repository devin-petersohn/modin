import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import ray
import ray.util.iter as iter
from ray.util.sgd.torch.torch_dataset import TorchMLDataset
from ray.util.sgd.torch.torch_trainer import TorchTrainer
from ray.util.sgd.torch.training_operator import TrainingOperator


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class ModinDataLoader(TorchMLDataset):
    def get_local_partitions(self):
        return self.get_shard(self.world_rank)

    def get_any_partitions(self, i):
        return self.get_shard(i)


def train_linear_sgd(df, model_class=Net, iterations=10, num_steps=100, config=None):
    parts = df._query_compiler._modin_frame._partitions.squeeze()
    it = iter.from_items([part.oid for part in parts], num_shards=16).for_each(
        lambda x: ray.get(x)
    )
    ds = ModinDataLoader.from_parallel_iter(it, False)
    torch_ds = ds.to_torch(feature_columns=["a"], label_column="b")

    def make_train_operator(ds: ModinDataLoader):
        class TrainOperator(TrainingOperator):
            model = model_class()

            def setup(self, config):
                optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=config.get("lr", 1e-2)
                )
                loss = torch.nn.MSELoss()
                batch_size = config.get("batch_size", 32)
                train_data = ds.get_shard(self.world_rank, shuffle=False)
                train_loader = DataLoader(train_data, batch_size=batch_size)
                self.model, self.optimizer, self.criterion = self.register(
                    models=self.model, optimizers=optimizer, criterion=loss
                )
                self.register_data(train_loader=train_loader, validation_loader=None)

        return TrainOperator

    if config is None:
        config = {}
    trainer = TorchTrainer(
        num_workers=4,
        training_operator_cls=make_train_operator(torch_ds),
        add_dist_sampler=False,
        config=config,
    )
    model = None
    for i in range(iterations):
        trainer.train(num_steps=num_steps)
        model = trainer.get_model()
    return model
