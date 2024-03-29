import torch
import torch.utils.data as data
from utils.visual_utils import pil_image_trans
from PIL import Image


class ImLingStructInfoDataset(data.Dataset):
    """ This class represents a dataset that states, for each image, different facts on the linguistic structure of its
        corresponding human annotations (captions, frames, etc., depends on the external dataset on which this dataset
        was built).
    """

    def __init__(self, sample_list, image_path_finder):
        super(ImLingStructInfoDataset, self).__init__()
        self.sample_list = sample_list
        self.image_path_finder = image_path_finder

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_struct_data = self.sample_list[idx]

        # Load image
        image_id = item_struct_data[0]
        image_obj = Image.open(self.image_path_finder.get_image_path(image_id))
        orig_image_size = image_obj.size
        image_tensor = pil_image_trans(image_obj)

        sample = {
            'image_id': image_id,
            'image': image_tensor,
            'orig_image_size': orig_image_size,
            'struct_info': item_struct_data[1],
        }

        return sample
