import torch.nn as nn


class ThreeLayerModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ThreeLayerModel, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.ReLU()

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation2 = nn.ReLU()

        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.activation3 = nn.ReLU()


    def forward(self, x):
        out = self.layer1(x)
        out = self.activation1(out)

        out = self.layer2(out)
        out = self.activation2(out)

        out = self.layer3(out)
        out = self.activation3(out)

        return out

    #three-layer network with nn.sequential
    model2 = nn.Sequential(
        nn.Linear(30, 100),
        nn.ReLU(),
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 1),
        nn.ReLU()
    )














