import os
import torch
from loggable_object import LoggableObject
from utils.general_utils import model_file_suffix, config_file_suffix

from model_src.image_linguistic_structural_info_classifiers.resnet_linguistic_structural_info_classifier \
    import ResNetLingStructInfoClassifier
from model_src.image_linguistic_structural_info_classifiers.clip_linguistic_structural_info_classifier \
    import CLIPLingStructInfoClassifier


class ModelFactory(LoggableObject):
    def __init__(self, indent):
        super(ModelFactory, self).__init__(indent)

    @staticmethod
    def create_model(model_config, model_dir, model_name):
        if model_config.backbone_model == 'resnet50':
            return ResNetLingStructInfoClassifier(model_config, model_dir, model_name)
        elif model_config.backbone_model == 'clip':
            return CLIPLingStructInfoClassifier(model_config, model_dir, model_name)

    def load_model(self, model_dir, model_name):
        config_file_path = os.path.join(model_dir, model_name + config_file_suffix)
        if not os.path.isfile(config_file_path):
            self.log_print('Couldn\'t find model "' + str(model_name) + '" in directory ' + str(model_dir))
            assert False

        config = torch.load(config_file_path)
        model_file_path = os.path.join(model_dir, model_name + model_file_suffix)
        model = torch.load(model_file_path)

        return model, config
