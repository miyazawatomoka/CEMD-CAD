from torch import nn, optim
from torch.utils.data import DataLoader
from model import VAE
import config


class VAETrainer:
    def __init__(self, dataset):
        self.model = VAE()
        if config.USE_GPU:
            self.model.cuda()
        self,dataset = dataset
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.train_loader = DataLoader(self.dataset(train=True))
        self.test_loader = DataLoader(self.dataset(train=False))

    def train(self):
        pass

    def test(self):
        pass
