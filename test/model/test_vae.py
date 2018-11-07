from unittest import TestCase
from model.vae import VAE


class TestVae(TestCase):
    def test_new(self):
        model = VAE()
        self.assertNotEqual(model, None)
