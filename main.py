from utils.general_utils import init_entry_point, log_print, project_root_dir, default_model_name
from model_src.model_config import ModelConfig
from executors.trainers.bp_trainer import BackpropagationTrainer
from executors.trainers.offline_trainer import OfflineTrainer
from dataset_builders.dataset_builder import DatasetBuilder
from dataset_builders.dataset_builder_creator import create_dataset_builder
from dataset_builders.concatenated_dataset_builder import ConcatenatedDatasetBuilder
from dataset_builders.single_dataset_builders.aggregated_dataset_builder import AggregatedDatasetBuilder
from dataset_list import language_dataset_list, translated_only_datasets, get_orig_dataset_to_configs
import os
import sys
import argparse


parser = argparse.ArgumentParser(description='Train and evaluate a multimodal word learning model.')
parser.add_argument('--write_to_log', action='store_true', default=False, dest='write_to_log',
                    help='redirect output to a log file')
parser.add_argument('--datasets_dir', type=str, default=os.path.join('..', 'datasets'), dest='datasets_dir',
                    help='the path to the datasets dir')
parser.add_argument('--language', type=str, default=None, dest='language',
                    help='the language of the used dataset')
parser.add_argument('--struct_property', type=str, dest='struct_property',
                    help='the linguistic structural property to be examined')
parser.add_argument('--dataset', type=str, default=None, dest='dataset',
                    help='the name of the used dataset')
parser.add_argument('--pretraining_method', type=str, default='image_net', dest='pretraining_method',
                    help='the pre-training method for the visual backbone model')
parser.add_argument('--classifier', type=str, default='neural', dest='classifier',
                    help='the type of classifier')
parser.add_argument('--svm_kernel', type=str, default='rbf', dest='svm_kernel',
                    help='the type of kernel if an SVM classifier is used')
parser.add_argument('--standardize_data', action='store_true', default=False, dest='standardize_data',
                    help='standardize data if an offline training method is used')
parser.add_argument('--classifier_layer_size', nargs="+", type=int, default=[], dest='classifier_layer_size',
                    help='the list of sizes of classifier layers if a neural classifier is used')
parser.add_argument('--classifier_activation_func', type=str, default='relu', dest='classifier_activation_func',
                    help='the activation function if a neural classifier is used')
parser.add_argument('--use_batch_norm', action='store_true', default=False, dest='use_batch_norm',
                    help='use batch normalization if a neural classifier is used')
parser.add_argument('--translated', action='store_true', default=False, dest='translated',
                    help='use translated captions')
parser.add_argument('--multilingual', action='store_true', default=False, dest='multilingual',
                    help='train the classifier on all languages combined')
parser.add_argument('--delete_model', action='store_true', default=False, dest='delete_model',
                    help='delete the created model at the end of training')
parser.add_argument('--dump_captions', action='store_true', default=False, dest='dump_captions',
                    help='only dump the captions of the dataset and exit')
args = parser.parse_args()
write_to_log = args.write_to_log
datasets_dir = args.datasets_dir
user_defined_language = args.language
struct_property = args.struct_property
dataset_name = args.dataset
pretraining_method = args.pretraining_method
classifier_name = args.classifier
svm_kernel = args.svm_kernel
standardize_data = args.standardize_data
classifier_layer_size = args.classifier_layer_size
classifier_activation_func = args.classifier_activation_func
use_batch_norm = args.use_batch_norm
translated = args.translated
multilingual = args.multilingual
delete_model = args.delete_model
dump_captions = args.dump_captions

DatasetBuilder.set_datasets_dir(datasets_dir)


def get_dataset_builder(cur_language, cur_dataset_name, cur_struct_property):
    if cur_language is None:
        # Create a joint builder of all languages
        orig_dataset_to_configs = get_orig_dataset_to_configs()
        external_builder_list = []
        for orig_dataset_name, configs in orig_dataset_to_configs.items():
            # First, filter translated configs
            filtered_configs = []
            for config in configs:
                if config[2]:
                    continue
                filtered_configs.append(config)
            # Now, create dataset builders
            builder_list = [get_dataset_builder(config[1], config[0], cur_struct_property)
                            for config in filtered_configs]
            agg_builder = AggregatedDatasetBuilder(config[0], builder_list, cur_struct_property, 1)
            external_builder_list.append(agg_builder)
        builder = ConcatenatedDatasetBuilder(external_builder_list, cur_struct_property, 1)
    else:
        # We have a specific language
        if cur_dataset_name is None:
            # Create a joint builder of all datasets of this language
            dataset_names = [x for x in language_dataset_list if x[0] == cur_language and x[2] == translated][0][1]
            builder_list = [create_dataset_builder(x, cur_struct_property, cur_language, translated)
                            for x in dataset_names]
            builder = ConcatenatedDatasetBuilder(builder_list, cur_struct_property, 1)
        else:
            # Create a build for a specific dataset
            builder = create_dataset_builder(
                cur_dataset_name, cur_struct_property, cur_language, translated
            )

    return builder


def prepare_train(cur_language, cur_dataset_name, cur_struct_property, should_write_to_log, indent):
    function_name = 'prepare_train'
    timestamp = init_entry_point(should_write_to_log)

    model_config = ModelConfig(
        struct_property=cur_struct_property,
        pretraining_method=pretraining_method,
        classifier=classifier_name,
        svm_kernel=svm_kernel,
        classifier_layer_size=classifier_layer_size,
        classifier_activation_func=classifier_activation_func,
        use_batch_norm=use_batch_norm,
        standardize_data=standardize_data
    )

    log_print(function_name, indent, str(model_config))
    log_print(function_name, indent, f'Dataset: {cur_dataset_name}, language: {cur_language}')

    log_print(function_name, indent, 'Generating datasets...')
    dataset_builder = get_dataset_builder(cur_language, cur_dataset_name, cur_struct_property)
    if dump_captions:
        dataset_builder.dump_captions()
        sys.exit(0)

    # Training set
    training_set = dataset_builder.build_dataset('train')
    log_print(function_name, indent, f'Training sample num: {len(training_set)}')

    # Test set
    test_set = dataset_builder.build_dataset('val')
    log_print(function_name, indent, f'Test sample num: {len(test_set)}')
    log_print(function_name, indent, 'datasets generated')

    return timestamp, training_set, test_set, model_config


def do_train(training_set, test_set, model_config, timestamp, indent):
    function_name = 'do_train'
    log_print(function_name, indent, 'Training model...')
    model_root_dir = os.path.join(project_root_dir, timestamp)
    if classifier_name == 'neural':
        trainer = BackpropagationTrainer(model_root_dir, training_set, test_set, 50, model_config, 1)
    elif classifier_name in ['svm', 'random_forest', 'xgboost']:
        trainer = OfflineTrainer(model_root_dir, training_set, test_set, 50, model_config, 1)
    else:
        log_print(function_name, indent, f'Classifier {classifier_name} not implemented. Stopping!')
        assert False
    trainer.run()
    log_print(function_name, indent, 'Finished training model')

    if delete_model:
        model_path = os.path.join(timestamp, default_model_name + '.mdl')
        os.remove(model_path)
        best_model_path = os.path.join(timestamp, default_model_name + '_best.mdl')
        os.remove(best_model_path)


def main(should_write_to_log):
    function_name = 'main'
    indent = 0

    if dataset_name in translated_only_datasets and (not translated):
        log_print(function_name, 0, f'Dataset {dataset_name} is only translated.'
                                    f' Please use the --translated flag. Stopping!')
        assert False

    # Traverse all linguistic properties
    if struct_property is None:
        struct_properties = ['negation', 'transitivity', 'root_pos', 'numbers', 'passive']
    else:
        struct_properties = [struct_property]
    for cur_struct_property in struct_properties:
        # Traverse all languages
        if user_defined_language is None:
            assert (not translated)
            # Find all languages
            languages = list(set([x[0] for x in language_dataset_list if not x[2]]))

            # Negation and passive are currently not implemented for Japanese
            if cur_struct_property in ['negation', 'passive']:
                languages = [x for x in languages if x != 'Japanese']
        else:
            languages = [user_defined_language]

        if multilingual:
            # Create a joint dataset from all languages
            assert user_defined_language is None
            timestamp, training_set, test_set, model_config = \
                prepare_train(None, dataset_name, cur_struct_property, should_write_to_log, indent)
            do_train(training_set, test_set, model_config, timestamp, indent)
        else:
            for cur_language in languages:
                timestamp, training_set, test_set, model_config = \
                    prepare_train(cur_language, dataset_name, cur_struct_property, should_write_to_log, indent)

                do_train(training_set, test_set, model_config, timestamp, indent)


main(write_to_log)
