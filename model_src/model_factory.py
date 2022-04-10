import os
import torch
from loggable_object import LoggableObject
from utils.general_utils import model_file_suffix, config_file_suffix

from model_src.image_linguistic_info_classifiers.neural_classifier import ImLingInfoNeuralClassifier
from model_src.image_linguistic_info_classifiers.svm_classifier import ImLingInfoSVMClassifier


class ModelFactory(LoggableObject):
    def __init__(self, indent):
        super(ModelFactory, self).__init__(indent)

    @staticmethod
    def create_model(model_config, model_dir, model_name):
        if model_config.classifier == 'neural':
            return ImLingInfoNeuralClassifier(model_config, model_dir, model_name)
        elif model_config.classifier == 'svm':
            return ImLingInfoSVMClassifier(model_config, model_dir, model_name)

    def load_model(self, model_dir, model_name):
        config_file_path = os.path.join(model_dir, model_name + config_file_suffix)
        if not os.path.isfile(config_file_path):
            self.log_print('Couldn\'t find model "' + str(model_name) + '" in directory ' + str(model_dir))
            assert False

        config = torch.load(config_file_path)
        model_file_path = os.path.join(model_dir, model_name + model_file_suffix)
        model = torch.load(model_file_path)
        model.set_dump_path(model_dir, model_name)

        return model, config
