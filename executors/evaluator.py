import torch
import torch.utils.data as data

from utils.general_utils import for_loop_with_reports
from executors.executor import Executor
from model_src.model_factory import ModelFactory

BATCH_REPORT_NUM = 100


class Evaluator(Executor):

    def __init__(self, test_set, loaded_model_dir, loaded_model_name, indent):
        super().__init__(indent)

        self.test_set = test_set

        model_factory = ModelFactory(self.indent + 1)
        self.model, self.model_config = model_factory.load_model(loaded_model_dir, loaded_model_name)

        self.correct_count = 0
        self.overall_count = 0

    """ The core function: evaluate the model given a batch of samples. """

    def evaluate_on_batch(self, index, sampled_batch, print_info):
        # Load data
        image_tensor = sampled_batch['image'].to(self.device)
        labels = sampled_batch['struct_info'].to(self.device)

        output = self.model(image_tensor)
        predictions = output.argmax(dim=1)

        label_num = labels.shape[0]
        incorrect_num = torch.sum((predictions + labels) % 2)
        correct_num = label_num - incorrect_num

        self.correct_count += correct_num
        self.overall_count += label_num

    """ Evaluate on the test set; This is the entry point of this class. """

    def evaluate(self):
        dataloader = data.DataLoader(self.test_set, batch_size=50, shuffle=False)

        checkpoint_len = BATCH_REPORT_NUM
        self.increment_indent()
        for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                              self.evaluate_on_batch, self.progress_report)
        self.decrement_indent()

        accuracy = self.correct_count/self.overall_count
        return accuracy
