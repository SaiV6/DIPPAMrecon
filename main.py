import argparse
from image_processing import readImg, constructPlot
from dip_network import dip_reconstruction
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Deep Image Prior Image Reconstruction")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--iterations", type=int, default=400, help="Number of iterations for DIP")
    parser.add_argument("--alpha", type=float, default=0.84, help="Alpha value for the loss function")
    parser.add_argument("--num_refinement_passes", type=int, default=3, help="Number of refinement passes")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save_path", type=str, default="reconstructed_image.png", help="Path to save the reconstructed image")

    args = parser.parse_args()

    # Load the image
    im_path = args.image
    rgb_image = Image.open(im_path)

    # Perform reconstruction with user-specified parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming dip_reconstruction returns a PIL image or a NumPy array
    reconstruction_result = dip_reconstruction(rgb_image, args.iterations, args.alpha, device, args.num_refinement_passes, args.accumulation_steps)

    # Display the reconstructed image
    plt.imshow(reconstruction_result)
    plt.title("Reconstructed Image")
    plt.axis('off')  # Hide axis
    plt.show()

    # Save the reconstructed image
    if isinstance(reconstruction_result, np.ndarray):
        # Convert NumPy array to PIL Image and save if the result is a NumPy array
        Image.fromarray(reconstruction_result.astype('uint8')).save(args.save_path)
    else:
        # Directly save if the result is already a PIL Image
        reconstruction_result.save(args.save_path)
    print(f"Reconstructed image saved to {args.save_path}")

if __name__ == "__main__":
    main()
