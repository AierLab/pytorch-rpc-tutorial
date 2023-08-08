import torch
import torch.nn as nn


class SimpleResnet(nn.Module):
    def __init__(self):
        super(SimpleResnet, self).__init__()

        # Define the neural network layers
        self.layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Second convolutional layer
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Third convolutional layer
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),  # Adaptive pooling layer to ensure fixed size output
            # Flatten the tensor for the fully connected layer
            nn.Flatten(),
            # Final fully connected layer
            nn.Linear(64, 10),
        )

    def forward(self, x):
        """
        Forward pass through the neural network layers.

        Args:
        - x (torch.Tensor): Input tensor to the neural network.

        Returns:
        - torch.Tensor: Output tensor after passing through the layers.
        """
        return self.layers(x)
