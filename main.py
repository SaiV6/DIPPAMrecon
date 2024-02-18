import argparse
from image_processing import readImg, constructPlot
from dip_network import dip_reconstruction
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import display

def main():
    parser = argparse.ArgumentParser(description="Deep Image Prior Image Reconstruction")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    # Load the image
    im_path = args.image
    rgb_image = Image.open(im_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    im, im_gt, im_masked, im_mask, im_bicubic, factor, im_bilinear, im_lanczos = readImg(rgb_image)

    data = im
    title = ['', 'Mask', 'Sparsely-sampled', 'Fully-sampled']
    constructPlot(data, title, fsize=(15, 5))
    plt.show()

    # Perform Deep Image Prior reconstruction with iterative refinement
    reconstructed_image = dip_reconstruction(im_mask, im_masked, im_gt, im, device)

    reconstructed_image2 = reconstructed_image * 255

    # Assuming reconstructed_image2 is your numpy array
    result_rgb = np.transpose(reconstructed_image2, (1, 2, 0))

    # Convert the numpy array to an image
    image_to_display = Image.fromarray(result_rgb.astype(np.uint8))

    result_rgb2 = result_rgb.astype(np.uint8)
    #result_rgb = np.dstack((reconstructed_image2[:, :, 0], reconstructed_image2[:, :, 1], reconstructed_image2[:, :, 2]))

    #image_to_display = Image.fromarray(result_rgb)

    #reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

    # Convert the numpy array to an image
    #image_to_show = Image.fromarray(reconstructed_image[2])
    #result_rgb = result_rgb / np.max(result_rgb)
    print(np.max(result_rgb))
    plt.figure
    plt.imshow(result_rgb2)
    plt.show()
    display(Image.fromarray(result_rgb.astype(np.uint8)))



if __name__ == "__main__":
    main()