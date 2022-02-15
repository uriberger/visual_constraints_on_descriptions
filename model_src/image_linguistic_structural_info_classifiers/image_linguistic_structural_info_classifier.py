import abc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general_utils import model_file_suffix, config_file_suffix
from utils.visual_utils import wanted_image_size


class ImLingStructInfoClassifier(nn.Module):

    """ This class predicts linguistic structural info of image descriptions, given only the images.
        It is based on a backbone visual model with a classification head for each predicted property.
    """

    def __init__(self, config, model_dir, model_name):
        super().__init__()

        self.config = config

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.backbone_model = self.generate_backbone_model()

        dummy_input = torch.zeros(1, 3, wanted_image_size[0], wanted_image_size)
        dummy_output = self.backbone_model_inference(dummy_input)
        backbone_output_size = dummy_output.shape[1]

        self.classification_heads = []
        for struct_property in config.struct_properties:
            if struct_property == 'passive':
                fc = nn.Linear(backbone_output_size, 2)
                sm = nn.Softmax(dim=1)
                self.classification_heads.append(nn.Sequential(fc, F.relu, sm))
            elif struct_property == 'empty_frame_slots_num':
                fc = nn.Linear(backbone_output_size, 6)
                sm = nn.Softmax(dim=1)
                self.classification_heads.append(nn.Sequential(fc, F.relu, sm))

        self.dump_path = os.path.join(model_dir, model_name)

    def forward(self, x):
        if self.config.freeze_backbone:
            with torch.no_grad():
                x = self.backbone_model_inference(x)
        else:
            x = self.backbone_model_inference(x)
        res = []
        for classification_head in self.classification_heads:
            res.append(classification_head(x))
        return res

    # Dumping and loading utilities

    """ Dump model and configuration to an external file.
        If a suffix is provided, add it to the end of the name of the dumped file.
    """

    def dump(self, suffix=None):
        old_dump_path = self.dump_path
        if suffix is not None:
            self.dump_path += '_' + suffix

        self.dump_config()
        self.dump_model()

        self.dump_path = old_dump_path

    """ Get the path to which the configuration will be dumped. """

    def get_config_path(self):
        return self.dump_path + config_file_suffix

    """ Dump the configuration to an external file. """

    def dump_config(self):
        torch.save(self.config, self.get_config_path())

    """ Get the path to which the model will be dumped. """

    def get_model_path(self):
        return self.dump_path + model_file_suffix

    """ Dump the model to an external file. """

    def dump_underlying_model(self):
        torch.save(self, self.get_model_path())

    # Abstract methods (to be implemented by inheritors)

    @abc.abstractmethod
    def generate_backbone_model(self):
        return

    @abc.abstractmethod
    def backbone_model_inference(self, input_tensor):
        return
