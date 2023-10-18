import torch.optim as optim
import torch.nn as nn

from executors.trainers.trainer import Trainer
from model_src.model_factory import ModelFactory


class BackpropagationTrainer(Trainer):

    def __init__(self, model_root_dir, training_set, test_set, batch_size, model_config, indent,
                 loaded_model_dir=None, loaded_model_name=None):
        super(BackpropagationTrainer, self).__init__(
            model_root_dir, training_set, test_set, batch_size, model_config, indent
        )

        model_factory = ModelFactory(self.indent + 1)
        if loaded_model_dir is None:
            # We train a brand new model
            self.model = model_factory.create_model(model_config, self.model_dir, self.model_name)
        else:
            # We continue training an existing model
            self.model_dir = loaded_model_dir
            self.model_name = loaded_model_name
            self.model, self.model_config = model_factory.load_model(self.model_dir, self.model_name)

        self.optimizer = optim.Adam(self.model.internal_classifier.parameters(), lr=model_config.learning_rate)
        self.criterion = self.get_criterion(model_config.struct_property)

        self.running_loss = 0.0
        self.metric_history = []

    @staticmethod
    def get_criterion(struct_property):
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
        metric = self.evaluate_current_model()
        self.decrement_indent()
        self.metric_history.append(metric)

        # If this is the best model, save it
        self.dump_best_model_if_needed()

        # We stop training when we get a result that doesn't improve for 5 consecutive epochs
        epoch_window_len = 5
        should_continue = (len(self.metric_history) < epoch_window_len or
                           max(self.metric_history[-epoch_window_len:]) != self.metric_history[-epoch_window_len])

        return should_continue

    def dump_best_model_if_needed(self):
        if self.metric_history[-1] == max(self.metric_history):
            # Current model is the best model so far
            self.model.dump('best')

    """ The core function: train the model given a batch of samples. """

    def train_on_batch(self, index, sampled_batch, print_info):
        # Load data
        image_tensor = sampled_batch['image'].to(self.device)
        labels = sampled_batch['struct_info'].to(self.device).float()

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        output = self.model.inference(image_tensor)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()

        # print statistics
        self.running_loss += loss.item()
        if print_info:
            self.log_print(f'[{self.epoch_ind}, {index}] loss: {self.running_loss / self.batch_report_num:.3f}')
            self.running_loss = 0.0

    """ Train on the training set """

    def train(self):
        should_continue = True
        epoch_ind = 0
        while should_continue:
            epoch_ind += 1
            self.log_print('Starting epoch ' + str(epoch_ind))
            self.epoch_ind = epoch_ind
            self.traverse_training_set(self.train_on_batch)
            should_continue = self.post_epoch()
