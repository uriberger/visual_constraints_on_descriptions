from utils.general_utils import init_entry_point, log_print, project_root_dir
from model_src.model_config import ModelConfig
from executors.trainer import Trainer
from dataset_builders.dataset_builder import DatasetBuilder
from dataset_builders.dataset_builder_creator import create_dataset_builder
import os
import argparse


parser = argparse.ArgumentParser(description='Train and evaluate a multimodal word learning model.')
parser.add_argument('--write_to_log', action='store_true', default=False, dest='write_to_log',
                    help='redirect output to a log file')
parser.add_argument('--datasets_dir', type=str, default=os.path.join('..', 'datasets'), dest='datasets_dir',
                    help='the path to the datasets dir')
parser.add_argument('--language', type=str, default='English', dest='language',
                    help='the language of the used dataset')
parser.add_argument('--struct_property', type=str, dest='struct_property',
                    help='the linguistic structural property to be examined')
parser.add_argument('--dataset', type=str, dest='dataset',
                    help='the name of the used dataset')
args = parser.parse_args()
write_to_log = args.write_to_log
datasets_dir = args.datasets_dir
language = args.language
struct_property = args.struct_property
dataset_name = args.dataset

DatasetBuilder.set_datasets_dir(datasets_dir)


def main(should_write_to_log):
    function_name = 'main'
    timestamp = init_entry_point(should_write_to_log, language)

    model_config = ModelConfig(struct_property=struct_property)
    log_print(function_name, 0, str(model_config))

    log_print(function_name, 0, 'Generating datasets...')
    training_set_builder = create_dataset_builder(dataset_name, 'train', struct_property)
    training_set = training_set_builder.build_dataset()
    test_set_builder = create_dataset_builder(dataset_name, 'val', struct_property)
    test_set = test_set_builder.build_dataset()
    log_print(function_name, 0, 'datasets generated')

    log_print(function_name, 0, 'Training model...')
    model_root_dir = os.path.join(project_root_dir, timestamp)
    trainer = Trainer(model_root_dir, training_set, test_set, 5, 50, model_config, 1)
    trainer.train()
    log_print(function_name, 0, 'Finished training model')


main(write_to_log)
