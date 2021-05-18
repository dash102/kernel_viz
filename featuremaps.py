import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from numpy.random import default_rng
import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

rng = default_rng()

class StepImage():
    def __init__(self, orig_input, step_size=2, is_normalized=True,
               renorm=True, eps=30, norm_update='l2'):
        self.orig_input = orig_input
        if is_normalized:
            mean=[0.485, 0.456, 0.406]
            std= [0.229, 0.224, 0.225]
        else:
            mean=[0., 0., 0.]
            std= [1., 1., 1.]

        is_cuda = orig_input.is_cuda
        self.mean = torch.tensor(mean)[:, None, None]
        self.std = torch.tensor(std)[:, None, None]
        if is_cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        self.eps = eps
        self.renorm = renorm
        self.step_size = step_size
        self.norm_update = norm_update

    def project(self, x):
        diff = x - self.orig_input
        if self.renorm:
            diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        val_projected = self.orig_input + diff

        val_projected *= self.std
        val_projected += self.mean
        val_clamped = torch.clamp(val_projected, 0, 1)
        val_clamped -= self.mean
        val_clamped /= self.std
        return val_clamped

    # move one step in the direction of the neuron gradient
    def step(self, x, g):
        step_size = self.step_size
        # Scale g so that each element of the batch is at least norm 1
        if self.norm_update == 'l2':
            l = len(x.shape) - 1
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        else:
            g_norm = torch.torch.abs(g).mean()
        scaled_g = g / (g_norm + 1e-10)
        stepped = x + scaled_g * step_size
        projected = self.project(stepped)
        return projected

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def prepare_image(image_cv2, do_normalize=True):
    # Resize
    img = cv2.resize(image_cv2, (224, 224))
    img = img[:, :, ::-1].copy()
    # Convert to tensor
    tensor_img = transforms.functional.to_tensor(img)

    # Possibly normalize
    if do_normalize:
        tensor_img = normalize(tensor_img)
    # Put image in a batch
    batch_tensor_img = torch.unsqueeze(tensor_img, 0)

    # Put the image in the gpu
    if cuda_available:
        batch_tensor_img = batch_tensor_img.cuda()
    return batch_tensor_img

def UnNormalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]):
    std_arr = torch.tensor(std)[:, None, None]
    mean_arr = torch.tensor(mean)[:, None, None]
    def func(img):
        img = img.clone()
        img *= std_arr
        img += mean_arr
        return img
    return func

unnormalize = UnNormalize()

def obtain_image(tensor_img, do_normalize=True):
    tensor_img = tensor_img.cpu()
    if do_normalize:
        tensor_img = unnormalize(tensor_img)
    img = transforms.functional.to_pil_image((tensor_img.data))
    return img

# This function creates a function that gives the output of a given
# network at layer: layer_id.
def model_layer(model, layer_id, device):
    # These are the 4 sequential layers of resnet50
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    def forward(input):
        layers_used = layers[:(layer_id+1)]
        x = input.to(device)
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        for l in layers_used:
            x = l(x)
        return x
    return forward

# Given a net and layer_id, extract a feature map from a blank image (use random channel)
# Returns matplotlib fig
def extract_feature_map(net, layer_id, device):
    # blank white image
    img = torch.ones((1, 3, 224, 224)).to(device)
    # plt.imshow(img[0].permute(1, 2, 0))

    # for param in net.parameters():
    #     param.requires_grad = False

    batch_tensor = img.clone().requires_grad_(True)
    step = StepImage(img, step_size=0.1, renorm=False, norm_update='abs', is_normalized=False)

    net_l = model_layer(net, layer_id, device)

    for _ in tqdm(range(100)):
        logit = net_l(batch_tensor)
        out_channels = logit.shape[1]
        channel_id = rng.choice(out_channels, replace=False)

        loss = torch.norm(logit[0,channel_id,...], p=2)
        gradient, = torch.autograd.grad(loss, batch_tensor)
        batch_tensor = step.step(batch_tensor, gradient)

    # visualization
    # original_image = obtain_image(img[0, :], do_normalize=False)
    modified_image = obtain_image(batch_tensor[0, :], do_normalize=False)

    fig, ax = plt.subplots(figsize=(5, 5))
    # axs[0].set_title('Original random image')
    # axs[0].imshow(original_image)
    ax.set_title(f'Modified image (layer {layer_id}, channel {channel_id})')
    ax.imshow(modified_image)
    return fig
