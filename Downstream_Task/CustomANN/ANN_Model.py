import torch.nn as nn

from Config.Downstream_Task_Config import ANN


class CustomANN(nn.Module):
    """
    Artificial Neural Network for downstream seismic signal classification.
    """

    def __init__(self, input_dim, num_classes):
        super(CustomANN, self).__init__()

        self.network = nn.Sequential(
            # First Hidden Layer
            nn.Linear(input_dim, ANN["hidden_layer_layout"][0]),
            nn.BatchNorm1d(ANN["hidden_layer_layout"][0]),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Second Hidden Layer
            nn.Linear(ANN["hidden_layer_layout"][0], ANN["hidden_layer_layout"][1]),
            nn.BatchNorm1d(ANN["hidden_layer_layout"][1]),
            nn.ReLU(),
            nn.Dropout(0.2),

            # third Hidden Layer
            nn.Linear(ANN["hidden_layer_layout"][1], ANN["hidden_layer_layout"][2]),
            nn.BatchNorm1d(ANN["hidden_layer_layout"][2]),
            nn.ReLU(),
            nn.Dropout(0.2),

            # 4 Hidden Layer
            nn.Linear(ANN["hidden_layer_layout"][2], ANN["hidden_layer_layout"][3]),
            nn.BatchNorm1d(ANN["hidden_layer_layout"][3]),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output Layer
            nn.Linear(ANN["hidden_layer_layout"][3], num_classes)
        )

    def forward(self, x):
        return self.network(x)