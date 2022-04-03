import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import os

from utils.general_utils import for_loop_with_reports, default_model_name
from executors.executor import Executor
from executors.evaluator import Evaluator
from model_src.model_factory import ModelFactory

BATCH_REPORT_NUM = 250


class Trainer(Executor):

    def __init__(self, model_root_dir, training_set, test_set, epoch_num, batch_size, model_config, indent,
                 loaded_model_dir=None, loaded_model_name=None):
        super().__init__(indent)

        self.training_set = training_set
        self.test_set = test_set
        self.epoch_num = epoch_num
        self.batch_size = batch_size

        model_factory = ModelFactory(self.indent + 1)
        if loaded_model_dir is None:
            # We train a brand new model
            self.model_dir = model_root_dir
            self.model_name = default_model_name
            self.model = model_factory.create_model(model_config, self.model_dir, self.model_name)
            self.model_config = model_config
        else:
            # We continue training an existing model
            self.model_dir = loaded_model_dir
            self.model_name = loaded_model_name
            self.model, self.model_config = model_factory.load_model(self.model_dir, self.model_name)

        self.optimizer = optim.Adam(self.model.parameters(), lr=model_config.learning_rate)
        self.criterion = self.get_criterion(model_config.struct_property)

        self.running_loss = 0.0
        self.accuracy_history = []

    @staticmethod
    def get_criterion(struct_property):
        # For now all struct properties use cross entropy loss
        return nn.CrossEntropyLoss()

    """ Actions that should be performed at the beginning of training. """

    def pre_training(self):
        self.model.dump()

    """ Actions that should be performed at the end of training. """

    def post_training(self):
        self.model.dump()

    """ Actions that should be performed at the end of each training epoch. """

    def post_epoch(self):
        self.model.dump()

        self.log_print('Evaluating after finishing the epoch...')
        self.increment_indent()
        accuracy = self.evaluate_current_model()
        self.decrement_indent()
        self.accuracy_history.append(accuracy)

        # If this is the best model, save it
        self.dump_best_model_if_needed()

    def evaluate_current_model(self):
        evaluator = Evaluator(self.test_set, self.model_dir, self.model_name, self.indent + 1)
        accuracy = evaluator.evaluate()
        return accuracy

    def dump_best_model_if_needed(self):
        if self.accuracy_history[-1] == max(self.accuracy_history):
            # Current model is the best model so far
            self.model.dump('best')

    """ The core function: train the model given a batch of samples. """

    def train_on_batch(self, index, sampled_batch, print_info):
        # Load data
        image_tensor = sampled_batch['image'].to(self.device)
        labels = sampled_batch['struct_info'].to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        output = self.model(image_tensor)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()

        # print statistics
        self.running_loss += loss.item()
        if print_info:
            self.log_print(f'[{self.epoch_ind}, {index}] loss: {self.running_loss / BATCH_REPORT_NUM:.3f}')
            self.running_loss = 0.0

    """ Train on the training set; This is the entry point of this class. """

    def train(self):
        self.pre_training()

        for epoch_ind in range(self.epoch_num):
            self.log_print('Starting epoch ' + str(epoch_ind))
            self.epoch_ind = epoch_ind

            dataloader = data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)

            checkpoint_len = BATCH_REPORT_NUM
            self.increment_indent()
            for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                                  self.train_on_batch, self.progress_report)
            self.decrement_indent()

            self.post_epoch()

        self.post_training()
