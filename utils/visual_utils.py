import numpy as np
from PIL import Image
import torchvision.transforms as transforms


# All images are resized to a uniform size
wanted_image_size = (224, 224)
# Pixel values are normalized
mean_tuple = (0.48145466, 0.4578275, 0.40821073)
std_tuple = (0.26862954, 0.26130258, 0.27577711)
# A transformation from PIL Image to a tensor
pil_image_trans = transforms.Compose([
    transforms.Resize(wanted_image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_tuple, std_tuple)
])


def get_image_shape(image_path):
    """ Get the original image size given it's id. """
    image_obj = Image.open(image_path)
    return np.array(image_obj).shape
