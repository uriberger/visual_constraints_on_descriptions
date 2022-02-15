import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

from utils.general_utils import for_loop_with_reports, default_model_name
from executors.executor import Executor
from model_src.model_factory import ModelFactory


class Trainer(Executor):

    def __init__(self, model_root_dir, training_set, epoch_num, batch_size, model_config, indent,
                 loaded_model_dir=None, loaded_model_name=None):
        super().__init__(indent)

        self.training_set = training_set
        self.epoch_num = epoch_num
        self.batch_size = batch_size

        model_factory = ModelFactory(self.indent + 1)
        if loaded_model_dir is None:
            # We train a brand new model
            model_dir = model_root_dir
            model_name = default_model_name
            self.model = model_factory.create_model(model_config, model_dir, model_name)
            self.model_config = model_config
        else:
            # We continue training an existing model
            model_dir = loaded_model_dir
            model_name = loaded_model_name
            self.model, self.model_config = model_factory.load_model(model_dir, model_name)

        self.optimizer = optim.Adam(self.model.parameters(), lr=model_config.learning_rate)
        self.criteria = self.get_criteria(model_config.struct_properties)

        self.running_loss = {struct_property: 0.0 for struct_property in model_config.struct_properties}
        self.batch_count = {struct_property: 0 for struct_property in model_config.struct_properties}
        self.num_of_batch_for_report = 1000

    @staticmethod
    def get_criteria(struct_properties):
        criteria = {}
        for struct_property in struct_properties:
            if struct_property == 'passive':
                criterion = nn.CrossEntropyLoss()
            elif struct_property == 'empty_frame_slots_num':
                criterion = nn.CrossEntropyLoss()
            criteria[struct_property] = criterion

        return criteria

    """ Actions that should be performed at the beginning of training. """

    def pre_training(self):
        self.model.dump()

    """ Actions that should be performed at the end of training. """

    def post_training(self):
        self.dump_models()

    """ Actions that should be performed at the end of each training epoch. """

    def post_epoch(self):
        self.dump_models()

        self.log_print('Evaluating after finishing the epoch...')
        self.increment_indent()
        self.evaluate_current_model()
        self.decrement_indent()

        # Dump results into a csv file
        self.dump_results_to_csv()

        # If this is the best model, save it
        self.dump_best_model_if_needed()

    def evaluate_current_model(self):
        return

    def dump_results_to_csv(self):
        return

    def dump_best_model_if_needed(self):
        return

    """ The core function: train the model given a batch of samples. """

    def train_on_batch(self, index, sampled_batch, print_info):
        # Load data
        image_tensor = sampled_batch[0].to(self.device)
        strucy_property_to_labels = sampled_batch[1]

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(image_tensor)
        for struct_property, labels in strucy_property_to_labels.items():
            loss = self.criteria[struct_property](outputs, labels)
            loss.backward()
            self.optimizer.step()
            # Check if after this line the gradients of the optimizer changed
            # print statistics
            self.running_loss[struct_property] += loss.item()
            self.batch_count[struct_property] += 1
            if self.batch_count[struct_property] % self.num_of_batch_for_report == self.num_of_batch_for_report - 1:
                print(f'[{self.epoch_ind}, {self.batch_count[struct_property]}] loss: ' +
                      f'{self.running_loss[struct_property] / self.num_of_batch_for_report:.3f}')
                self.running_loss[struct_property] = 0.0

    """ Train on the training set; This is the entry point of this class. """

    def train(self):
        self.model.dump()

        for epoch_ind in range(self.epoch_num):
            self.log_print('Starting epoch ' + str(epoch_ind))
            self.epoch_ind = epoch_ind

            dataloader = data.DataLoader(self.training_set, batch_size=self.batch_size)

            checkpoint_len = 1000
            self.increment_indent()
            for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                                  self.train_on_batch, self.progress_report)
            self.decrement_indent()

            self.post_epoch()

        self.post_training()
