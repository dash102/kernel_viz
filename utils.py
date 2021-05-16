import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

rng = default_rng()

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

# normalize image (for matplotlib viewing purposes)
def norm(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min)

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# helper function to log to tensorboard
def plot_kernels_tensorboard(layer, num_kernels_to_sample):
    out_channels = layer.shape[0]
    in_channels = 1

    rows = (num_kernels_to_sample + 3)//4
    fig, axs = plt.subplots(rows, 4)
    fig.set_size_inches(in_channels, 1)

    random_kernel_i = rng.choice(out_channels, size=num_kernels_to_sample, replace=False)
    random_kernels = layer[random_kernel_i]

    for i in range(num_kernels_to_sample):
        #
        kernel = random_kernels[i].swapaxes(0, 2).swapaxes(0, 1)
        if kernel.shape[2] > 3:
            random_channel_i = rng.choice(kernel.shape[2])
            kernel = kernel[:, :, random_channel_i]
        axs[i//4][i%4].imshow(norm(kernel), cmap='gray')

        axs[i//4][i%4].axis(False)
    # clear up extra axes labels    
    for i in range(num_kernels_to_sample, 4 * rows):
        axs[i//4][i%4].axis(False)

    return fig
