from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def readImg(im, axis1_range=[200, 600], axis2_range=[300, 1400]):
    try:
        im_array = np.array(im)

        # Assign each channel to the designated variable
        im_mask = np.copy(im_array[:, :, 0])  # Binary image of the mask
        im_masked = np.copy(im_array[:, :, 1])
        im_gt = np.copy(im_array[:, :, 2])

        # Get the down-sampled image
        im_down = im_masked[im_mask != 0]
        [x_down, y_down] = np.where(im_mask != 0)
        x_down = np.unique(x_down)
        y_down = np.unique(y_down)
        im_down = im_down.reshape(len(x_down), len(y_down))
    except:
        im_array = np.array(im)
        # There will be data that the mask and the ground truth channel swap
        # Reassign each channel to the designated variable
        im_mask = np.copy(im_array[:, :, 2])  # Binary image of the mask
        im_masked = np.copy(im_array[:, :, 1])
        im_gt = np.copy(im_array[:, :, 0])
        im_down = im_masked[im_mask != 0]
        [x_down, y_down] = np.where(im_mask != 0)
        x_down = np.unique(x_down)
        y_down = np.unique(y_down)
        im_down = im_down.reshape(len(x_down), len(y_down))
        im = np.dstack((im_mask, im_masked, im_gt))

    # Convert the NumPy arrays to PIL Images
    im_down_pil = Image.fromarray(im_down)

    # Resize the down-sampled image to match the ground truth size
    im_bicubic = im_down_pil.resize((im_gt.shape[1], im_gt.shape[0]), resample=Image.BICUBIC)
    im_bilinear = im_down_pil.resize((im_gt.shape[1], im_gt.shape[0]), resample=Image.BILINEAR)
    im_lanczos = im_down_pil.resize((im_gt.shape[1], im_gt.shape[0]), resample=Image.LANCZOS)

    # Get interpolated factor
    factor = np.divide(im_gt.shape, (len(x_down), len(y_down)))
    factor = np.round(factor)
    factor = factor.astype(int)

    return im, im_gt, im_masked, im_mask, im_bicubic, factor, im_bilinear, im_lanczos

def constructPlot(data, title, fsize=(10, 6), cmax=None):
    """ Construct subplots based on the stacked matrices and its corresponded label
        NOTE: title[0] is the main title. title[i] is the """
    fig = plt.figure(figsize=fsize)
    for i in range(data.shape[2]):
        ax = fig.add_subplot(1, data.shape[2], i + 1)
        plt.imshow(np.rot90(data[:, :, i]),
                   interpolation='nearest', aspect='auto', cmap='gray')
        plt.axis('off')
        if cmax is not None:
            plt.clim(data[:, :, i].min(), cmax)
        ax.set_title(title[i + 1], fontsize=15)
