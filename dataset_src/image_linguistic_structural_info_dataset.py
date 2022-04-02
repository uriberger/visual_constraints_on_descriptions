import torch
import torch.utils.data as data
from utils.general_utils import get_image_id_to_prob
from utils.visual_utils import pil_image_trans
from PIL import Image


class ImLingStructInfoDataset(data.Dataset):
    """ This class represents a dataset that states, for each image, different facts on the linguistic structure of its
        corresponding human annotations (captions, frames, etc., depends on the external dataset on which this dataset
        was built).
    """

    def __init__(self, struct_data, image_path_finder, threshold):
        super(ImLingStructInfoDataset, self).__init__()
        self.struct_data = struct_data
        if threshold is None:
            self.threshold = self.find_threshold()
        else:
            self.threshold = threshold
        self.sample_list = self.generate_sample_list()
        self.image_path_finder = image_path_finder

    """ This class's input is a list of (image_id, val) where image ids are not unique and val is binary value
        representing whether a specific caption (related to the image id) expresses the relevant property.
        We want to:
        1. Convert this list to a mapping of image id -> probability of the relevant property, which is calculated as
        the proportion of captions expressing this property.
        2. Set a threshold for the probability so the final dataset will be binary.
        
        To choose the threshold we search for the value that, if chosen, will make the data be the closest to half 1
        half 0.
    """

    def find_threshold(self):
        # 1. Convert this list to a mapping of image id -> probability of the relevant property, which is calculated as
        #    the proportion of captions expressing this property.
        image_id_to_prob = get_image_id_to_prob(self.struct_data)

        # 2. Set a threshold for the probability so the final dataset will be binary.
        all_values = list(set(image_id_to_prob.values()))
        assert len(all_values) > 1
        val_to_count = {x: 0 for x in all_values}
        for val in image_id_to_prob.values():
            val_to_count[val] += 1
        val_to_count_list = sorted(val_to_count.items(), key=lambda x: x[0])
        current_count = 0
        cur_ind = -1
        total_count = len(image_id_to_prob)
        while current_count < total_count / 2:
            cur_ind += 1
            current_count += val_to_count_list[cur_ind][1]

        # Found the ind of the median, now check if this one is closer to half or the previous one
        current_ratio = current_count / total_count
        prev_ratio = (current_count - val_to_count_list[cur_ind][1]) / total_count
        if abs(0.5 - current_ratio) < abs(0.5 - prev_ratio):
            threshold = val_to_count_list[cur_ind][0]
        else:
            threshold = val_to_count_list[cur_ind - 1][0]

        return threshold

    def generate_sample_list(self):
        image_id_to_prob = get_image_id_to_prob(self.struct_data)
        return [
            (x[0], int(x[1] > self.threshold)) for x in image_id_to_prob.items()
        ]

    def __len__(self):
        return len(self.struct_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_struct_data = self.struct_data[idx]

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
