import torch

from executors.trainers.trainer import Trainer
from model_src.model_factory import ModelFactory
import numpy as np


class SVMTrainer(Trainer):

    def __init__(self, model_root_dir, training_set, test_set, batch_size, model_config, indent):
        super(SVMTrainer, self).__init__(model_root_dir, training_set, test_set, batch_size, model_config, indent)

        model_factory = ModelFactory(self.indent + 1)
        self.model = model_factory.create_model(model_config, self.model_dir, self.model_name)

        self.training_mat = None
        self.label_mat = None

    """ Actions that should be performed at the beginning of training. """

    def pre_training(self):
        return

    """ Actions that should be performed at the end of training. """

    def post_training(self):
        self.model.dump()
        self.log_print('Evaluating after finishing the training...')
        self.increment_indent()
        self.evaluate_current_model()
        self.decrement_indent()

    def generate_training_label_mat(self):
        sample_num = len(self.training_set)
        feature_num = self.model.get_backbone_output_size()

        self.training_mat = np.zeros((sample_num, feature_num))
        self.label_mat = np.zeros(sample_num)

        self.batch_start_ind = 0
        self.traverse_training_set(self.collect_from_batch)

    def collect_from_batch(self, index, sampled_batch, print_info):
        with torch.no_grad():
            # Load data
            image_tensor = sampled_batch['image'].to(self.device)
            labels = sampled_batch['struct_info'].to(self.device)

            extracted_features = self.model.backbone_model_inference(image_tensor)

        batch_size = len(labels)
        self.training_mat[self.batch_start_ind:self.batch_start_ind + batch_size, :] = extracted_features.cpu()
        self.label_mat[self.batch_start_ind:self.batch_start_ind + batch_size] = labels.cpu()
        self.batch_start_ind += batch_size

    def train(self):
        self.generate_training_label_mat()
        self.model.fit(self.training_mat, self.label_mat)
