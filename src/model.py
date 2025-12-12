import torch
import torch.nn as nn
import numpy as np


class FUNSR_Net(nn.Module):
    def __init__(self, hidden_dim=256, layers=8, skip=4):
        super().__init__()
        self.skip = skip
        self.layers = nn.ModuleList()

        # Input layer (x,y,z) -> hidden
        self.layers.append(nn.Linear(3, hidden_dim))

        # Hidden layers
        for i in range(layers - 2):
            if i + 1 == skip:
                self.layers.append(nn.Linear(hidden_dim + 3, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer -> SDF scalar
        self.last_layer = nn.Linear(hidden_dim, 1)

        # Activation (Softplus beta=100 như bài báo)
        self.act = nn.Softplus(beta=100)

        # Geometric Initialization [cite: 1242]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, np.sqrt(2) / np.sqrt(m.out_features))
                nn.init.constant_(m.bias, 0.0)
        # Init bias lớp cuối = -0.5 để bắt đầu với hình cầu r=0.5
        nn.init.constant_(self.last_layer.bias, -0.5)

    def forward(self, x):
        inp = x.clone()
        h = x
        for i, layer in enumerate(self.layers):
            if i == self.skip:
                h = torch.cat([h, inp], dim=-1)
            h = self.act(layer(h))
        return self.last_layer(h)

    def get_sdf_and_gradient(self, x):
        x.requires_grad_(True)
        sdf = self.forward(x)
        # Tính đạo hàm bậc 1 (Gradient)
        grad = torch.autograd.grad(
            sdf, x,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return sdf, grad


class Discriminator(nn.Module):
    """Adversarial Discriminator (OSC-ADL)"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)