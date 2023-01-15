# Author: Keanu Spies (keanuspies@gmail.com)
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

batch_sz = 64
dtype = torch.float
device = torch.device("mps")

###### HELPER FUNCTIONS ######

def tensor2image(image):
    """
    Converts a tensor image to a numpy uint8 (imshow-able) version. 
    params: 
        - image : singular (pytorch) tensor image 
    returns:
        - image : numpy array
    """

    image = image.cpu().detach()
    image = image.permute((1, 2, 0))
    return image.numpy()

def display_batch(batch, gray=False, save_fn = ''):
    """
    Displays a batch images. 
    params: 
     - batch : (batch_sz, c, h, w) tensor of images to display
    """ 
    plt.figure(figsize=(5, 5))
    for i in range(batch.shape[0]):
        plt.subplot(int(math.sqrt(batch_sz)), int(math.sqrt(batch_sz)), i + 1)
        # convert to numpy image
        img = tensor2image(batch[i])
        if gray:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.axis("off")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    if save_fn: 
        plt.savefig(save_fn)
    plt.show()


###### DATASET CLASS ######

class FilterDataset(ImageFolder):
    """
    Image dataset class with targets of a chosen filter type. 
    Can either be a 'sobel' filter or a 'random' filter.
    """
    def __init__(self, root, kernel, kernel_type='single', normalize=True, transform=None):
        super().__init__(root, transform=transform)
        """
        params: 
        - root: string directory of images
        - kernel (3x3 array): which kernel to use
        - kernel_type (string): either 'single' (in this case just the uni-directional kernel) or 
                              'double' (corresponds to the gradient magnitude - sobel operation)
        - normalize (bool): whether or not to normalize the output of the filter. Note for 'double' this is ignored.
        - transform (torch.transform): what image transform to use
        """

        self.filter_kernel =  kernel
        self.filter_kernel = torch.tensor(self.filter_kernel, dtype=torch.float32).to(device)
        self.kernel_type = kernel_type
        self.normalize = normalize

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)

        img = img.to(device)
        
        # apply the kernel filter to the image
        if self.kernel_type == 'double':
            kernel = self.filter_kernel.unsqueeze(0).unsqueeze(0) # x
            kernel_t = self.filter_kernel.T.unsqueeze(0).unsqueeze(0) # y
            
            gray_img = transforms.Grayscale()(img)
            
            # horizontal filter (Gx)
            f_img_x = F.conv2d(gray_img, kernel, stride=1, padding=1)
            # vertical filter (Gy)
            f_img_y = F.conv2d(gray_img, kernel_t, stride=1, padding=1)

            # the opperation is the magnitude of the Gx and Gy outputs
            out = torch.cat([f_img_x, f_img_y], axis = 0)

        elif self.kernel_type == 'single':

            # apply on the x direction filter
            kernel = self.filter_kernel.unsqueeze(0).unsqueeze(0)
            gray_img = transforms.Grayscale()(img)
            out = F.conv2d(gray_img, kernel, stride=1, padding=1)
            
        else:
            print('Please enter a valid kernel type (\'single\' or  \'double\')')
            return None

        # normalization
        if self.normalize:
            out = torch.mul(out, out)
            out = torch.sum(out, axis=0, keepdim=True)
            out = torch.sqrt(out)
            
        return img, out