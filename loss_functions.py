import torch
import torch.nn.functional as F
from torch.autograd import Variable

def l1_loss(P, alpha=0.84):
    return torch.mean(torch.abs(P))


def gaussian_filter(x, sigma):
    # Assuming x is a 3D tensor [channels, height, width]

    # Calculate the size of the Gaussian kernel
    kernel_size = int(4 * sigma) * 2 + 1

    # Create a 2D Gaussian kernel with the same number of channels as x
    gaussian_kernel = torch.ones(x.shape[0], 1, kernel_size, kernel_size).to(x.device) / (kernel_size ** 2)

    # Expand dimensions to match the number of channels in input tensor
    gaussian_kernel = gaussian_kernel.expand(-1, x.shape[0], -1, -1)

    # Apply 2D convolution
    filtered_x = F.conv2d(x.unsqueeze(0), gaussian_kernel, padding=int(4 * sigma)).squeeze(0)

    return filtered_x

def ms_ssim_loss(P, M=5, alpha=1.0):
    # Assume P is a 3D tensor [channels, height, width]

    # Add a batch dimension
    P = P.unsqueeze(0)

    levels = []
    for j in range(1, M + 1):
        scale_factor = 2 ** (M - j)
        P_resized = F.interpolate(P, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        levels.append(gaussian_filter(P_resized.squeeze(), sigma=1.5))

    lM = levels[-1]
    csj = levels[:-1]

    ms_ssim = torch.prod(lM ** alpha * torch.prod(csj ** alpha, dim=0))
    return 1 - ms_ssim

def combined_loss(P, alpha=0.84, sigma_values=[0.5, 1, 2, 4, 8]):
    l1 = l1_loss(P, alpha=alpha)
    ms_ssim = ms_ssim_loss(P, M=len(sigma_values), alpha=1.0)

    # Gaussian filters for MS-SSIM
    gaussian_filters = [gaussian_filter(P, sigma) for sigma in sigma_values]

    # Combine MS-SSIM and l1 using the specified weights
    LMix = alpha * ms_ssim + (1 - alpha) * torch.prod(gaussian_filters) * l1

    return LMix