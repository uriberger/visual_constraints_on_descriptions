from collections import defaultdict
import numpy as np


# Utility functions


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator/denominator


def get_image_id_to_count(struct_data):
    """ Given a list of (image_id, val)- the struct_data input- where image ids are not unique and val is a binary
        value, get 2 mappings: one from image id to the number of its instances in the list, and one from image id to
        the number of its instances in the list where the val was 1.
    """
    image_id_to_count = defaultdict(int)
    image_id_to_pos_count = defaultdict(int)
    for image_id, has_num in struct_data:
        image_id_to_count[image_id] += 1
        image_id_to_pos_count[image_id] += has_num
    return image_id_to_count, image_id_to_pos_count


def get_image_id_to_prob(struct_data):
    """ Given a list of (image_id, val)- the struct_data input- where image ids are not unique and val is a binary
        value, get a mapping from image id to the fraction of its instances in the list where the val was 1.
    """
    image_id_to_count, image_id_to_pos_count = get_image_id_to_count(struct_data)
    image_id_to_prob = {x: image_id_to_pos_count[x] / image_id_to_count[x] for x in image_id_to_count.keys()}
    return image_id_to_prob


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


# Data analysis functions


def get_class_prob_list(struct_data, gt_class_data, class_mapping):
    """ Given a list of (image_id, val)- the struct_data input- where image ids are not unique and val is a binary
        value, and a mapping from image id to a list of gt classes in this image, do the following:
        1. For each image id, calculate the fraction of its instances in the struct_data list where the val was 1.
        2. For each gt class, calculate the mean fraction from 1, averaged across all image ids relevant for this class.
        3. Convert each class ind to a name using the provided class mapping.
    """
    # Preprocessing: make sure struct_data and gt_class data contain the same imade ids
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
        range(len(class_mapping))}

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
    all_image_ids_dict = {x: True for x in all_image_ids}
    image_id_to_prob1 = {x[0]: x[1] for x in image_id_to_prob1.items() if x[0] in all_image_ids_dict}
    image_id_to_prob2 = {x[0]: x[1] for x in image_id_to_prob2.items() if x[0] in all_image_ids_dict}

    # Make sure all image ids in the first dictionary are in the second dict as well and vice versa
    assert len(image_id_to_prob1) == len(image_id_to_prob2)
    assert len([x for x in image_id_to_prob1.keys() if x in image_id_to_prob2]) == len(image_id_to_prob1)

    # Now that the image ids are the same, if we sort according to image id we can compare
    vals1 = [x[1] for x in image_id_to_prob1.items().sorted(key=lambda x: x[0])]
    vals2 = [x[1] for x in image_id_to_prob2.items().sorted(key=lambda x: x[0])]

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


def get_mean_values_across_datasets(struct_datas_list):
    """ Given a list of lists of (image_id, val)- the struct_datas_list input- where image ids are not unique and val is
        a binary value, and we assume the struct_data in the list share image ids, do the following:
        1. In each list, for each image id, calculate the fraction of its instances in the struct_data list where the
        val was 1.
        2. For each image id, compute the mean of fractions from 1 across different struct datas.
    """
    # Preprocessing: make sure struct_data and gt_class data contain the same imade ids
    unique_image_ids_list = [set([x[0] for x in struct_data]) for struct_data in struct_datas_list]
    all_image_ids = list(set.intersection(*unique_image_ids_list))

    # Now find the sum of fractions (aka probabilities) across all struct datas
    image_id_to_prob_sum = defaultdict(int)
    for struct_data in struct_datas_list:
        image_id_to_prob = get_image_id_to_prob(struct_data)
        for image_id in all_image_ids:
            image_id_to_prob_sum[image_id] += image_id_to_prob[image_id]

    # Finally, because we want the mean and not the sum, divide by the number of struct datas
    image_id_to_mean_prob = {x[0]: x[1]/len(struct_datas_list) for x in image_id_to_prob_sum.items()}

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
    image_id_to_prob1 = get_image_id_to_prob(struct_data1)
    image_id_to_prob2 = get_image_id_to_prob(struct_data2)
    all_image_ids = [x for x in image_id_to_prob1.keys() if x in image_id_to_prob2]
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
