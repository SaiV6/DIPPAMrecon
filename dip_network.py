import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

class DIPNetwork(nn.Module):
    def __init__(self, input_channels=3, noise_std=0.2):
        super(DIPNetwork, self).__init__()
        self.noise_std = noise_std

        # Expanded architecture with more layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Inject random noise
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        x = self.encoder(x)
        x = self.decoder(x)
        return x

def dip_reconstruction(mask, sparsely_sampled, fully_sampled, im, device, num_iterations=400, alpha=0.84, num_refinement_passes=3, accumulation_steps=4):
    # Convert PIL Images to PyTorch tensors
    transform = transforms.ToTensor()
    mask = transform(mask).to(device)
    sparsely_sampled = transform(sparsely_sampled).to(device)
    fully_sampled = transform(fully_sampled).to(device)

    # Combine the inputs into a single tensor
    input_tensor = torch.cat([mask, sparsely_sampled, fully_sampled], dim=0)

    # Initialize the DIP network
    dip_network = DIPNetwork().to(device)

    # Apply weight initialization
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)

    # Apply the initialization to the network
    dip_network.apply(weights_init)

    # Use Adam optimizer for simplicity
    optimizer = optim.Adam(dip_network.parameters(), lr=0.0001)

    # Define the loss function
    criterion = nn.MSELoss()
    huber = nn.SmoothL1Loss()

    # Record the start time
    start_time = time.time()

    loss_vals = []

    # Iterative refinement loop
    for refinement_pass in range(num_refinement_passes):
        print(f"Refinement Pass {refinement_pass + 1}/{num_refinement_passes}")
        for iteration in range(num_iterations):
            # Forward pass
            output = dip_network(input_tensor)

            # Compute the loss
            #loss2 = combined_loss(output - fully_sampled)
            loss = huber(output, fully_sampled)

            # Backward pass and optimization
            #loss2 = loss2 / accumulation_steps
            loss = loss / accumulation_steps

            #loss2.backward(retain_graph=True)  # Retain graph for the next backward pass
            loss.backward()

            if (iteration + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_vals.append(loss.item())

            print(f"Iteration {iteration}, Loss: {loss.item()}")
            #CombinedLoss: {loss2.item()}")

      # Adjust noise level for the next refinement pass
        dip_network.noise_std /= 2.0

    plt.plot(range(num_iterations * num_refinement_passes), loss_vals, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Record the end time
    end_time = time.time()

    # Calculate and print the total time taken
    total_time = end_time - start_time
    print(f"Total time taken for reconstruction: {total_time:.2f} seconds")

    # Extract the reconstructed image
    reconstructed_image = output.detach().cpu().numpy()

    return reconstructed_image
