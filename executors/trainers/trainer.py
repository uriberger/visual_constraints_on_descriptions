import abc

from utils.general_utils import default_model_name, for_loop_with_reports
from executors.executor import Executor
from executors.evaluator import Evaluator

import torch.utils.data as data


BATCH_REPORT_NUM = 250


class Trainer(Executor):

    def __init__(self, model_root_dir, training_set, test_set, batch_size, model_config, indent):
        super(Trainer, self).__init__(indent)

        self.training_set = training_set
        self.test_set = test_set

        self.model_dir = model_root_dir
        self.model_name = default_model_name
        self.model_config = model_config

        self.batch_size = batch_size
        self.batch_report_num = BATCH_REPORT_NUM

    """ Actions that should be performed at the beginning of training. """

    @abc.abstractmethod
    def pre_training(self):
        return

    """ Actions that should be performed at the end of training. """

    @abc.abstractmethod
    def post_training(self):
        return

    def evaluate_current_model(self):
        if self.model_config.struct_property.startswith('length_'):
            evaluation_mode = 'ordered_classification'
        else:
            evaluation_mode = 'binary_classification' # We also have 'regression' but this is not currently supported
        evaluator = Evaluator(evaluation_mode, self.test_set, self.model_dir, self.model_name, self.indent + 1)
        metric = evaluator.evaluate()
        return metric

    def traverse_training_set(self, batch_func):
        dataloader = data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)

        checkpoint_len = self.batch_report_num
        self.increment_indent()
        for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                              batch_func, self.progress_report)
        self.decrement_indent()

    @abc.abstractmethod
    def train(self):
        return

    """ This is the entry point of this class. """
    def run(self):
        self.pre_training()
        self.train()
        self.post_training()
