from utils.general_utils import init_entry_point, log_print, project_root_dir
from model_src.model_config import ModelConfig
from executors.trainers.bp_trainer import BackpropagationTrainer
from executors.trainers.offline_trainer import OfflineTrainer
from dataset_builders.dataset_builder import DatasetBuilder
from dataset_builders.dataset_builder_creator import create_dataset_builder
from dataset_builders.concatenated_dataset_builder import ConcatenatedDatasetBuilder
from dataset_list import language_dataset_list, translated_only_datasets
import os
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
parser.add_argument('--classifier_layer_size', nargs="+", type=int, default=[], dest='classifier_layer_size',
                    help='the list of sizes of classifier layers if a neural classifier is used')
parser.add_argument('--classifier_activation_func', type=str, default='relu', dest='classifier_activation_func',
                    help='the activation function if a neural classifier is used')
parser.add_argument('--use_batch_norm', action='store_true', default=False, dest='use_batch_norm',
                    help='use batch normalization if a neural classifier is used')
parser.add_argument('--translated', action='store_true', default=False, dest='translated',
                    help='use translated captions')
parser.add_argument('--dump_captions', action='store_true', default=False, dest='dump_captions',
                    help='only dump the captions of the dataset and exit')
parser.add_argument('--balanced_training_set', action='store_true', default=False, dest='balanced_training_set',
                    help='balance the number of samples for each label in the training set')
parser.add_argument('--balanced_test_set', action='store_true', default=False, dest='balanced_test_set',
                    help='balance the number of samples for each label in the test set')
args = parser.parse_args()
write_to_log = args.write_to_log
datasets_dir = args.datasets_dir
language = args.language
struct_property = args.struct_property
dataset_name = args.dataset
pretraining_method = args.pretraining_method
classifier_name = args.classifier
svm_kernel = args.svm_kernel
classifier_layer_size = args.classifier_layer_size
classifier_activation_func = args.classifier_activation_func
use_batch_norm = args.use_batch_norm
translated = args.translated
dump_captions = args.dump_captions
balanced_training_set = args.balanced_training_set
balanced_test_set = args.balanced_test_set

DatasetBuilder.set_datasets_dir(datasets_dir)


def get_dataset_builder(cur_language, data_split_str):
    if dataset_name is None:
        dataset_names = [x for x in language_dataset_list if x[0] == cur_language and x[2] == translated][0][1]
        builder_list = [create_dataset_builder(x, data_split_str, struct_property, translated)
                        for x in dataset_names]
        builder = ConcatenatedDatasetBuilder(builder_list, struct_property, 1)
    else:
        builder = create_dataset_builder(dataset_name, data_split_str, struct_property, translated)

    return builder


def main(should_write_to_log):
    function_name = 'main'

    model_config = ModelConfig(
        struct_property=struct_property,
        pretraining_method=pretraining_method,
        classifier=classifier_name,
        svm_kernel=svm_kernel,
        classifier_layer_size=classifier_layer_size,
        classifier_activation_func=classifier_activation_func,
        use_batch_norm=use_batch_norm
    )

    if dataset_name in translated_only_datasets and (not translated):
        log_print(function_name, 0, f'Dataset {dataset_name} is only translated.'
                                    f' Please use the --translated flag. Stopping!')
        assert False

    if language is None:
        assert (not translated)
        # Find all languages
        languages = list(set([x[0] for x in language_dataset_list if not x[2]]))
    else:
        languages = [language]
    for cur_language in languages:
        timestamp = init_entry_point(should_write_to_log, cur_language)
        indent = 0
        log_print(function_name, indent, str(model_config))
        log_print(function_name, indent, f'Dataset: {dataset_name}, language: {cur_language}')

        log_print(function_name, indent, 'Generating datasets...')
        training_set_builder = get_dataset_builder(cur_language, 'train')
        test_set_builder = get_dataset_builder(cur_language, 'val')
        if dump_captions:
            training_set_builder.dump_captions()
            test_set_builder.dump_captions()
            return

        # Training set
        training_set = training_set_builder.build_dataset()
        training_set.generate_sample_list()
        threshold = training_set.get_threshold()
        training_label_to_data_samples = training_set.find_samples_for_labels()
        training_label_to_sample_num = {x[0]: len(x[1]) for x in training_label_to_data_samples.items()}
        log_print(function_name, indent, f'Training sample num per label: {training_label_to_sample_num}')
        if balanced_training_set:
            log_print(function_name, indent, 'Balancing training data')
            training_set.balance_data()
            log_print(function_name, indent, f'After balancing, training data contains {len(training_set.sample_list)} samples')

        # Test set
        test_set = test_set_builder.build_dataset()
        test_set.generate_sample_list(threshold)
        test_label_to_data_samples = test_set.find_samples_for_labels()
        test_label_to_sample_num = {x[0]: len(x[1]) for x in test_label_to_data_samples.items()}
        log_print(function_name, indent, f'Test sample num per label: {test_label_to_sample_num}')
        if balanced_test_set:
            log_print(function_name, indent, 'Balancing test data')
            test_set.balance_data()
            log_print(function_name, indent, f'After balancing, test data contains {len(test_set.sample_list)} samples')
        log_print(function_name, indent, 'datasets generated')

        log_print(function_name, indent, 'Training model...')
        model_root_dir = os.path.join(project_root_dir, timestamp)
        if classifier_name == 'neural':
            trainer = BackpropagationTrainer(model_root_dir, training_set, test_set, 20, 50, model_config, 1)
        elif classifier_name in ['svm', 'random_forest', 'xgboost']:
            trainer = OfflineTrainer(model_root_dir, training_set, test_set, 50, model_config, 1)
        else:
            log_print(function_name, indent, f'Classifier {classifier_name} not implemented. Stopping!')
            assert False
        trainer.run()
        log_print(function_name, indent, 'Finished training model')


main(write_to_log)
