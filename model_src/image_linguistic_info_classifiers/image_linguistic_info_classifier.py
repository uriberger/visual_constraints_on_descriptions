import abc
import os
import torch
import torch.nn as nn

from utils.general_utils import model_file_suffix, config_file_suffix
from utils.visual_utils import wanted_image_size

import clip
import torchvision.models as models


class ImLingInfoClassifier(nn.Module):

    """ This class predicts linguistic structural info of image descriptions, given only the images. """

    def __init__(self, config, model_dir, model_name):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.config = config

        if config.struct_property == 'passive':
            self.output_size = 2
        elif config.struct_property == 'empty_frame_slots_num':
            self.output_size = 6
        elif config.struct_property == 'transitivity':
            self.output_size = 2
        elif config.struct_property == 'negation':
            self.output_size = 2
        elif config.struct_property == 'numbers':
            self.output_size = 2
        elif config.struct_property == 'root_pos':
            self.output_size = 2

        self.set_dump_path(model_dir, model_name)
        self.backbone_model = None
        self.backbone_output_size = None

    # Dumping and loading utilities

    def set_dump_path(self, model_dir, model_name):
        self.dump_path = os.path.join(model_dir, model_name)

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

    def dump_model(self):
        torch.save(self, self.get_model_path())

    def generate_backbone_model(self):
        if self.config.backbone_model == 'resnet50':
            return models.resnet50(pretrained=self.config.freeze_backbone).to(self.device)
        elif self.config.backbone_model == 'clip':
            return clip.load('RN50', self.device)[0]

    def get_backbone_model(self):
        if self.backbone_model is None:
            self.backbone_model = self.generate_backbone_model()
        return self.backbone_model

    def backbone_model_inference(self, input_tensor):
        if self.config.backbone_model == 'resnet50':
            return self.get_backbone_model()(input_tensor)
        elif self.config.backbone_model == 'clip':
            return self.get_backbone_model().encode_image(input_tensor).float()

    def get_backbone_output_size(self):
        if self.backbone_output_size is None:
            dummy_input = torch.zeros(1, 3, wanted_image_size[0], wanted_image_size[1]).to(self.device)
            dummy_output = self.backbone_model_inference(dummy_input)
            self.backbone_output_size = dummy_output.shape[1]
        return self.backbone_output_size

    @abc.abstractmethod
    def predict(self, image_tensor):
        return
