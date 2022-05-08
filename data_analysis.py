from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from dataset_builders.dataset_builder_creator import create_dataset_builder
from dataset_builders.concatenated_dataset_builder import ConcatenatedDatasetBuilder
from utils.general_utils import safe_divide, get_image_id_to_prob, get_image_id_to_count
from dataset_list import language_dataset_list, get_orig_dataset_to_configs


# Utility functions


def get_class_to_image_list(gt_class_data):
    """ Given a mapping from image id to a list of gt classes in this image, get the opposite mapping- from gt class to
        the list of image ids in which this class is instantiated.
    """
    class_to_image_list = defaultdict(list)
    for image_id, gt_class_list in gt_class_data.items():
        unique_gt_class_list = list(set(gt_class_list))
        for gt_class in unique_gt_class_list:
            class_to_image_list[gt_class].append(image_id)
    return class_to_image_list


def get_dataset_builder(language, dataset_name, struct_property, translated):
    dataset_builder = create_dataset_builder(dataset_name, struct_property, language, translated)
    return dataset_builder


def get_dataset(language, dataset_name, struct_property, translated):
    concat_builder = get_dataset_builder(language, dataset_name, struct_property, translated)
    dataset = concat_builder.build_dataset('all')
    return dataset


def get_intersection_image_ids(struct_datas):
    """ Get a list of image ids that exist in all struct datas. """
    unique_image_ids_list = [set([x[0] for x in struct_data]) for struct_data in struct_datas]
    all_image_ids = list(set.intersection(*unique_image_ids_list))
    return all_image_ids


# Data analysis functions


def get_class_prob_list(struct_data, gt_class_data, class_mapping):
    """ Given a list of (image_id, val)- the struct_data input- where image ids are not unique and val is a binary
        value, and a mapping from image id to a list of gt classes in this image, do the following:
        1. For each image id, calculate the fraction of its instances in the struct_data list where the val was 1.
        2. For each gt class, calculate the mean fraction from 1, averaged across all image ids relevant for this class.
        3. Convert each class ind to a name using the provided class mapping.
    """
    # Preprocessing: make sure struct_data and gt_class data contain the same image ids
    all_image_ids = list(set([x[0] for x in struct_data if x[0] in gt_class_data]))
    gt_class_data = {x: gt_class_data[x] for x in all_image_ids}
    all_image_ids_dict = {x: True for x in all_image_ids}
    struct_data = [x for x in struct_data if x[0] in all_image_ids_dict]

    class_to_image_list = get_class_to_image_list(gt_class_data)

    # 1. For each image id, calculate the fraction of its instances in the struct_data list where the val was 1.
    image_id_to_prob = get_image_id_to_prob(struct_data)

    # 2. For each gt class, calculate the mean fraction from 1, averaged across all image ids relevant for this class.
    class_to_prob_mean = {
        i: safe_divide(sum([image_id_to_prob[x] for x in class_to_image_list[i]]), len(class_to_image_list[i])) for i in
        range(len(class_mapping)) if len(class_to_image_list[i]) > 0}

    class_prob_list = list(class_to_prob_mean.items())
    class_prob_list.sort(reverse=True, key=lambda x: x[1])
    # 3. Convert each class ind to a name using the provided class mapping.
    class_prob_list = [(class_mapping[x[0]], x[1]) for x in class_prob_list]

    return class_prob_list


def get_bbox_count_dist(struct_data, gt_bbox_data):
    """ Given a list of (image_id, val)- the struct_data input- where image ids are not unique and val is a binary
        value, and a mapping from image id to a list of gt bounding boxes in this image, do the following:
        1. For each image id, calculate the fraction of its instances in the struct_data list where the val was 1.
        2. For each number of gt bboxes, calculate the mean fraction from 1, averaged across all image ids where this is
        the number of bounding boxes.
    """
    # Preprocessing: make sure struct_data and gt_class data contain the same imade ids
    all_image_ids = list(set([x[0] for x in struct_data if x[0] in gt_bbox_data]))
    gt_bbox_data = {x: gt_bbox_data[x] for x in all_image_ids}
    image_id_to_bbox_num = {x[0]: len(x[1]) for x in gt_bbox_data.items()}
    all_image_ids_dict = {x: True for x in all_image_ids}
    struct_data = [x for x in struct_data if x[0] in all_image_ids_dict]

    # 1. For each image id, calculate the fraction of its instances in the struct_data list where the val was 1.
    image_id_to_prob = get_image_id_to_prob(struct_data)

    max_bbox_num = max(image_id_to_bbox_num.values())
    bbox_val_to_image_list = get_class_to_image_list({x[0]: [x[1]] for x in image_id_to_bbox_num.items()})
    # 2. For each number of gt bboxes, calculate the mean fraction from 1, averaged across all image ids where this is
    #    the number of bounding boxes.
    bbox_val_to_prob_mean = {
        i: safe_divide(sum([image_id_to_prob[x] for x in bbox_val_to_image_list[i]]), len(bbox_val_to_image_list[i]))
        for i in range(max_bbox_num + 1)}

    bbox_val_prob_list = list(bbox_val_to_prob_mean.items())
    bbox_val_prob_list.sort(key=lambda x: x[0])
    bbox_val_prob_list = [x[1] for x in bbox_val_prob_list]

    return bbox_val_prob_list


def get_vals_agreement(struct_data1, struct_data2):
    """ Given two lists of (image_id, val)- the struct_data1/2 input- where image ids are not unique and val is a binary
        value, calculate the pearson correlation coefficient of the two lists.
        To do this, we need the lists to be sorted by image ids.
    """
    # First make sure both lists contain the same image ids
    image_id_to_prob1 = get_image_id_to_prob(struct_data1)
    image_id_to_prob2 = get_image_id_to_prob(struct_data2)
    all_image_ids = [x for x in image_id_to_prob1.keys() if x in image_id_to_prob2]
    print('\t\tImage num in comparison: ' + str(len(all_image_ids)))
    all_image_ids_dict = {x: True for x in all_image_ids}
    image_id_to_prob1 = {x[0]: x[1] for x in image_id_to_prob1.items() if x[0] in all_image_ids_dict}
    image_id_to_prob2 = {x[0]: x[1] for x in image_id_to_prob2.items() if x[0] in all_image_ids_dict}

    # Make sure all image ids in the first dictionary are in the second dict as well and vice versa
    assert len(image_id_to_prob1) == len(image_id_to_prob2)
    assert len([x for x in image_id_to_prob1.keys() if x in image_id_to_prob2]) == len(image_id_to_prob1)

    # Now that the image ids are the same, if we sort according to image id we can compare
    vals1 = [x[1] for x in sorted(list(image_id_to_prob1.items()), key=lambda x: x[0])]
    vals2 = [x[1] for x in sorted(list(image_id_to_prob2.items()), key=lambda x: x[0])]

    np_arr = np.array([vals1, vals2])
    pearson_corr = np.corrcoef(np_arr)[0, 1]

    return pearson_corr


def get_mean_val(struct_datas_list):
    """ Given a list of lists of (image_id, val)- the struct_datas_list input- where image ids are not unique and val is
        a binary value, do the following:
        1. In each list, for each image id, calculate the fraction of its instances in the struct_data list where the
        val was 1.
        2. Calculate the mean fraction over all images in all lists.
    """
    total_sum = 0
    total_count = 0
    for struct_data in struct_datas_list:
        image_id_to_prob = get_image_id_to_prob(struct_data)
        total_sum += sum(image_id_to_prob.values())
        total_count += len(image_id_to_prob)

    return total_sum/total_count


def get_mean_values_across_datasets(struct_datas_list, languages, aggregate_per_language):
    """ Given a list of lists of (image_id, val)- the struct_datas_list input- where image ids are not unique and val is
        a binary value, and we assume the struct_data in the list share image ids, and the corresponding list of
        languages, do the following:
        1. In each list, for each image id, calculate the fraction of its instances in the struct_data list where the
        val was 1.
        2. For each image id, compute the mean of fractions from 1 across different struct datas.
    """
    if not aggregate_per_language:
        languages = ['all_languages']*len(struct_datas_list)

    # Preprocessing: make sure struct_data and gt_class data contain the same imade ids
    all_image_ids = get_intersection_image_ids(struct_datas_list)
    all_image_ids_dict = {x: True for x in all_image_ids}

    # Now find the sum of fractions (aka probabilities) across all struct datas, for each language
    unique_languages = list(set(languages))
    language_image_id_to_prob_sum = {x: defaultdict(int) for x in unique_languages}
    language_image_id_to_prob_count = {x: defaultdict(int) for x in unique_languages}
    for i in range(len(struct_datas_list)):
        struct_data = struct_datas_list[i]
        language = languages[i]
        for image_id, val in struct_data:
            if image_id in all_image_ids_dict:
                language_image_id_to_prob_sum[language][image_id] += val
                language_image_id_to_prob_count[language][image_id] += 1

    # Next, because we want the mean and not the sum, divide the sum by the count
    language_image_id_to_prob_mean = {
        x[0]: {
            y[0]: y[1]/language_image_id_to_prob_count[x[0]][y[0]]
            for y in x[1].items()
        }
        for x in language_image_id_to_prob_sum.items()
    }

    # Finally, aggregate across all languages and divide by the number of languages
    image_id_to_prob_sum = {x: 0 for x in all_image_ids}
    for cur_image_id_to_prob_mean in language_image_id_to_prob_mean.values():
        for image_id, prob_mean in cur_image_id_to_prob_mean.items():
            image_id_to_prob_sum[image_id] += prob_mean

    image_id_to_mean_prob = {x[0]: x[1]/len(unique_languages) for x in image_id_to_prob_sum.items()}

    return image_id_to_mean_prob


def get_extreme_non_agreement_image_ids(struct_data1, struct_data2):
    """ Given two lists of (image_id, val)- the struct_data1/2 input- where image ids are not unique and val is a binary
        value, do the following:
        1. For each image id in each struct data, calculate the fraction of its instances in the struct_data list where
        the val was 1.
        2. Find image ids with extreme values of opposite ends in the two struct datas (1 in the first and 0 in the
        second).
    """
    # First make sure both lists contain the same image ids
    image_id_to_count1 = get_image_id_to_count(struct_data1)[0]
    image_id_to_count2 = get_image_id_to_count(struct_data2)[0]
    image_id_to_prob1 = get_image_id_to_prob(struct_data1)
    image_id_to_prob2 = get_image_id_to_prob(struct_data2)
    all_image_ids = [x for x in image_id_to_prob1.keys() if x in image_id_to_prob2]

    # Filter all images without that maximal number of captions
    max_cap_num1 = max(image_id_to_count1.values())
    max_cap_num2 = max(image_id_to_count2.values())
    all_image_ids = [x for x in all_image_ids
                     if image_id_to_count1[x] == max_cap_num1 and image_id_to_count2[x] == max_cap_num2]

    all_image_ids_dict = {x: True for x in all_image_ids}
    image_id_to_prob1 = {x[0]: x[1] for x in image_id_to_prob1.items() if x[0] in all_image_ids_dict}
    image_id_to_prob2 = {x[0]: x[1] for x in image_id_to_prob2.items() if x[0] in all_image_ids_dict}

    # Make sure all image ids in the first dictionary are in the second dict as well and vice versa
    assert len(image_id_to_prob1) == len(image_id_to_prob2)
    assert len([x for x in image_id_to_prob1.keys() if x in image_id_to_prob2]) == len(image_id_to_prob1)

    high_val_in_1 = [x[0] for x in image_id_to_prob1.items() if x[1] == 1]
    extreme_list_high_in_1 = [x for x in high_val_in_1 if image_id_to_prob2[x] == 0]

    high_val_in_2 = [x[0] for x in image_id_to_prob2.items() if x[1] == 1]
    extreme_list_high_in_2 = [x for x in high_val_in_2 if image_id_to_prob1[x] == 0]

    return extreme_list_high_in_1, extreme_list_high_in_2


# Main function


def get_class_prob_list_for_config(language, dataset_name, struct_property, translated):
    builder = get_dataset_builder(language, dataset_name, struct_property, translated)
    dataset = builder.build_dataset('all')

    struct_data = dataset.struct_data
    gt_class_data = builder.get_gt_classes_data()
    class_mapping = builder.get_class_mapping()

    class_prob_list = get_class_prob_list(struct_data, gt_class_data, class_mapping)
    return class_prob_list


def get_bbox_dist_list_for_config(language, dataset_name, struct_property, translated):
    builder = get_dataset_builder(language, dataset_name, struct_property, translated)
    dataset = builder.build_dataset('all')

    struct_data = dataset.struct_data
    gt_bbox_data = builder.get_gt_bboxes_data()

    bbox_count_dist_list = get_bbox_count_dist(struct_data, gt_bbox_data)
    return bbox_count_dist_list


def generate_list_edges_str(input_list, edge_size):
    res = 'High:\n'
    for i in range(edge_size):
        if i > 0:
            res += ', '
        res += input_list[i][0] + ': ' + '{:.4f}'.format(input_list[i][1])
    res += '\nLow:\n'
    for i in range(edge_size):
        if i > 0:
            res += ', '
        res += input_list[-(i+1)][0] + ': ' + '{:.4f}'.format(input_list[-(i+1)][1])

    return res


def print_class_prob_lists(struct_property):
    english_coco_class_prob_list = \
        get_class_prob_list_for_config('English', 'COCO', struct_property, False)
    japanese_coco_class_prob_list = \
        get_class_prob_list_for_config('Japanese', 'STAIR-captions', struct_property, False)
    chinese_coco_class_prob_list = \
        get_class_prob_list_for_config('Chinese', 'coco-cn', struct_property, False)
    translated_chinese_coco_class_prob_list = \
        get_class_prob_list_for_config('Chinese', 'coco-cn', struct_property, True)
    translated_german_coco_class_prob_list = \
        get_class_prob_list_for_config('German', 'de_coco', struct_property, True)

    print('English coco class prob list:')
    print(generate_list_edges_str(english_coco_class_prob_list, 5))
    print('\nJapanese coco class prob list:')
    print(generate_list_edges_str(japanese_coco_class_prob_list, 5))
    print('\nChinese coco class prob list:')
    print(generate_list_edges_str(chinese_coco_class_prob_list, 5))
    print('\nTranslated Chinese coco class prob list:')
    print(generate_list_edges_str(translated_chinese_coco_class_prob_list, 5))
    print('\nTranslated German coco class prob list:')
    print(generate_list_edges_str(translated_german_coco_class_prob_list, 5))


def plot_bbox_dist_lists(struct_property):
    english_coco_bbox_dist_list = \
        get_bbox_dist_list_for_config('English', 'COCO', struct_property, False)
    japanese_coco_bbox_dist_list = \
        get_bbox_dist_list_for_config('Japanese', 'STAIR-captions', struct_property, False)
    chinese_coco_bbox_dist_list = \
        get_bbox_dist_list_for_config('Chinese', 'coco-cn', struct_property, False)
    translated_chinese_coco_bbox_dist_list = \
        get_bbox_dist_list_for_config('Chinese', 'coco-cn', struct_property, True)

    plt.plot(english_coco_bbox_dist_list, label='English')
    plt.plot(japanese_coco_bbox_dist_list, label='Japanese')
    plt.plot(chinese_coco_bbox_dist_list, label='Chinese')
    plt.plot(translated_chinese_coco_bbox_dist_list, label='Translated Chinese')

    plt.legend()
    plt.xlabel('Number of bounding boxes')
    plt.ylabel('Mean ' + struct_property + ' probability')
    plt.title('Mean ' + struct_property + ' probability as a function of bbox #')
    plt.show()


def print_language_agreement(struct_property, with_translated):
    orig_dataset_to_configs = get_orig_dataset_to_configs()

    for orig_dataset_name, configs in orig_dataset_to_configs.items():
        print(orig_dataset_name + ':')
        # First, filter configs if needed
        filtered_configs = []
        for config in configs:
            if (not with_translated) and config[2]:
                continue
            filtered_configs.append(config)

        # Next, generate data for each config
        struct_datas = []
        for dataset_name, language, translated in filtered_configs:
            dataset = get_dataset(language, dataset_name, struct_property, translated)
            struct_datas.append(dataset.struct_data)

        # Filter image ids so that each struct data will have the same image ids
        intersection_image_ids = get_intersection_image_ids(struct_datas)
        intersection_image_ids_dict = {x: True for x in intersection_image_ids}
        struct_datas = [[y for y in x if y[0] in intersection_image_ids_dict] for x in struct_datas]

        # Next, calculate agreement between each two datasets
        for i in range(len(filtered_configs)):
            for j in range(i+1, len(filtered_configs)):
                lang1 = filtered_configs[i][1]
                if filtered_configs[i][2]:
                    lang1 += '_translated'
                lang2 = filtered_configs[j][1]
                if filtered_configs[j][2]:
                    lang2 += '_translated'
                pearson_coef = get_vals_agreement(struct_datas[i], struct_datas[j])
                print('\t' + lang1 + ' and ' + lang2 + ' agreement: ' + '{:.4f}'.format(pearson_coef))


def print_language_mean_val(struct_property):
    for language, dataset_name_list, translated in language_dataset_list:
        for dataset_name in dataset_name_list:
            struct_datas = []
            dataset = get_dataset(language, dataset_name, struct_property, translated)
            struct_datas.append(dataset.struct_data)
        mean_val = get_mean_val(struct_datas)
        language_str = language
        if translated:
            language_str += '_translated'
        print(language_str + ': ' + '{:.4f}'.format(mean_val))


def print_consistently_extreme_image_ids(struct_property, aggregate_per_language):
    orig_dataset_to_configs = get_orig_dataset_to_configs()

    for orig_dataset_name, configs in orig_dataset_to_configs.items():
        print(orig_dataset_name + ':')
        # First, generate data for each config
        struct_datas = []
        languages = []
        for dataset_name, language, translated in configs:
            dataset = get_dataset(language, dataset_name, struct_property, translated)
            struct_datas.append(dataset.struct_data)
            language_name = language
            if translated:
                language_name += '_translated'
            languages.append(language_name)

        # Next, calculate mean across all datasets
        image_id_to_mean_prob = get_mean_values_across_datasets(struct_datas, languages, aggregate_per_language)
        image_id_mean_prob_list = sorted(list(image_id_to_mean_prob.items()), key=lambda x: x[1], reverse=True)
        image_id_mean_prob_list = [(str(x[0]), x[1]) for x in image_id_mean_prob_list]

        print(generate_list_edges_str(image_id_mean_prob_list, min(5, len(image_id_mean_prob_list))))


def print_extreme_non_agreement_image_ids(struct_property):
    orig_dataset_to_configs = get_orig_dataset_to_configs()

    for orig_dataset_name, configs in orig_dataset_to_configs.items():
        print(orig_dataset_name + ':')
        # First, generate data for each config
        all_languages = list(set([x[1] for x in configs]))
        language_to_dataset_builders = {lan: defaultdict(list) for lan in all_languages}
        for dataset_name, language, translated in configs:
            builder = get_dataset_builder(language, dataset_name, struct_property, translated)
            language_to_dataset_builders[language][translated].append(builder)

        # Join all builders to a single builder
        language_to_concat_builder = {
            x[0]: {
                y[0]:
                    ConcatenatedDatasetBuilder(y[1], struct_property, 1) for y in x[1].items()
            }
            for x in language_to_dataset_builders.items()
        }
        # Build dataset
        language_to_dataset = {}
        for language, translated_to_builder in language_to_concat_builder.items():
            for translated, builder in translated_to_builder.items():
                language_name = language
                if translated:
                    language_name += '_translated'
                language_to_dataset[language_name] = builder.build_dataset('all')
        # Build struct data list
        language_to_struct_data_temp = {x[0]: x[1].struct_data
                                        for x in language_to_dataset.items()}
        language_to_struct_data = {x[0]: [(language_to_dataset[x[0]].image_path_finder.new_to_orig_image_id(y[0])[0],
                                           y[1]) for y in x[1]]
                                   for x in language_to_struct_data_temp.items()}

        # Finally, search for cases of extreme differences
        my_list = list(language_to_struct_data.items())
        for i in range(len(my_list)):
            for j in range(i + 1, len(my_list)):
                extreme_list_high_in_1, extreme_list_high_in_2 = \
                    get_extreme_non_agreement_image_ids(my_list[i][1], my_list[j][1])
                if len(extreme_list_high_in_1) > 0:
                    list_for_print = extreme_list_high_in_1[:min(5, len(extreme_list_high_in_1))]
                    print(f'\tHigh in {my_list[i][0]}, low in {my_list[j][0]}: {list_for_print}')
                if len(extreme_list_high_in_2) > 0:
                    list_for_print = extreme_list_high_in_2[:min(5, len(extreme_list_high_in_2))]
                    print(f'\tHigh in {my_list[j][0]}, low in {my_list[i][0]}: {list_for_print}')


def plot_image_histogram(struct_property):
    all_language_vals = []

    # fig, ax = plt.subplots()
    # default_ticks = list(ax.get_yticks())
    # if min([x for x in default_ticks if x > 0]) >= 1000:
    #     y_labels = [str(int(x / 1000)) + 'k' for x in default_ticks]
    #     ax.set_yticks(ticks=default_ticks, labels=y_labels)
    # ax.set_title(struct_property + ' histogram')
    # ax.set_xlabel('Proportion of captions with the property')
    # ax.set_ylabel('Number of images')
    i = 0

    for language, dataset_list, translated in language_dataset_list:
        if translated:
            continue
        language_vals = []
        for dataset_name in dataset_list:
            builder = get_dataset_builder(language, dataset_name, struct_property, translated)
            image_id_to_prob = get_image_id_to_prob(builder.get_struct_data())
            language_vals += list(image_id_to_prob.values())
        all_language_vals += language_vals
        x_vals = sorted(list(set(language_vals)))
        count_dict = defaultdict(int)
        for val in language_vals:
            count_dict[val] += 1
        y_vals = [count_dict[x_val] for x_val in x_vals]

        # fig, ax = plt.subplots()
        # ax.bar(x_vals, y_vals, width=0.1)
        # default_ticks = list(ax.get_yticks())
        # if min([x for x in default_ticks if x > 0]) >= 1000:
        #     y_labels = [str(int(x / 1000)) + 'k' for x in default_ticks]
        #     ax.set_yticks(ticks=default_ticks, labels=y_labels)
        # ax.set_title(struct_property + ' histogram in ' + language)
        # ax.set_xlabel('Proportion of captions with the property')
        # ax.set_ylabel('Number of images')
        plt.bar([z - 0.05 + 0.025*i for z in x_vals], y_vals, width=0.025, label=language)
        i += 1
    plt.legend()
    plt.show()
    plt.clf()

    # All languages combined
    fig, ax = plt.subplots()

    x_vals = sorted(list(set(all_language_vals)))
    count_dict = defaultdict(int)
    for val in all_language_vals:
        count_dict[val] += 1
    y_vals = [count_dict[x_val] for x_val in x_vals]

    ax.bar(x_vals, y_vals, width=0.1)
    default_ticks = list(ax.get_yticks())
    y_labels = [str(int(x / 1000)) + 'k' for x in default_ticks]
    ax.set_yticks(ticks=default_ticks, labels=y_labels)
    ax.set_title(struct_property + ' histogram in all languages combined')
    ax.set_xlabel('Proportion of captions with the property')
    ax.set_ylabel('Number of images')
    plt.show()


def analyze(struct_property):
    # print_class_prob_lists(struct_property)
    # plot_bbox_dist_lists(struct_property)
    # print_language_agreement(struct_property, True)
    # print_language_agreement(struct_property, False)
    # print_language_mean_val(struct_property)
    # print_consistently_extreme_image_ids(struct_property, True)
    # print_consistently_extreme_image_ids(struct_property, False)
    # print_extreme_non_agreement_image_ids(struct_property)
    plot_image_histogram(struct_property)


analyze('numbers')
