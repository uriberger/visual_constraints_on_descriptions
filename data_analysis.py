from collections import defaultdict


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator/denominator


def get_image_id_to_count(struct_data):
    image_id_to_count = defaultdict(int)
    image_id_to_pos_count = defaultdict(int)
    for image_id, has_num in struct_data:
        image_id_to_count[image_id] += 1
        image_id_to_pos_count[image_id] += has_num
    return image_id_to_count, image_id_to_pos_count


def get_image_id_to_prob(struct_data):
    image_id_to_count, image_id_to_pos_count = get_image_id_to_count(struct_data)
    image_id_to_prob = {x: image_id_to_pos_count[x] / image_id_to_count[x] for x in image_id_to_count.keys()}
    return image_id_to_prob


def get_class_to_image_list(gt_class_data):
    class_to_image_list = defaultdict(list)
    for image_id, gt_class_list in gt_class_data.items():
        unique_gt_class_list = list(set(gt_class_list))
        for gt_class in unique_gt_class_list:
            class_to_image_list[gt_class].append(image_id)
    return class_to_image_list


def get_class_prob_list(struct_data, gt_class_data, class_mapping):
    all_image_ids = list(set([x[0] for x in struct_data if x[0] in gt_class_data]))
    gt_class_data = {x: gt_class_data[x] for x in all_image_ids}
    all_image_ids_dict = {x: True for x in all_image_ids}
    struct_data = [x for x in struct_data if x[0] in all_image_ids_dict]

    class_to_image_list = get_class_to_image_list(gt_class_data)

    image_id_to_prob = get_image_id_to_prob(struct_data)
    class_to_prob_mean = {
        i: safe_divide(sum([image_id_to_prob[x] for x in class_to_image_list[i]]), len(class_to_image_list[i])) for i in
        range(len(class_mapping))}

    class_prob_list = list(class_to_prob_mean.items())
    class_prob_list.sort(reverse=True, key=lambda x: x[1])
    class_prob_list = [(class_mapping[x[0]], x[1]) for x in class_prob_list]

    return class_prob_list


def get_bbox_count_dist(struct_data, gt_bbox_data):
    all_image_ids = list(set([x[0] for x in struct_data if x[0] in gt_bbox_data]))
    gt_bbox_data = {x: gt_bbox_data[x] for x in all_image_ids}
    image_id_to_bbox_num = {x[0]: len(x[1]) for x in gt_bbox_data.items()}
    all_image_ids_dict = {x: True for x in all_image_ids}
    struct_data = [x for x in struct_data if x[0] in all_image_ids_dict]

    image_id_to_prob = get_image_id_to_prob(struct_data)

    max_bbox_num = max(image_id_to_bbox_num.values())
    bbox_val_to_image_list = get_class_to_image_list({x[0]: [x[1]] for x in image_id_to_bbox_num.items()})
    bbox_val_to_prob_mean = {
        i: safe_divide(sum([image_id_to_prob[x] for x in bbox_val_to_image_list[i]]), len(bbox_val_to_image_list[i]))
        for i in range(max_bbox_num + 1)}

    bbox_val_prob_list = list(bbox_val_to_prob_mean.items())
    bbox_val_prob_list.sort(key=lambda x: x[0])
    bbox_val_prob_list = [x[1] for x in bbox_val_prob_list]

    return bbox_val_prob_list


def get_language_agreement(struct_data1, struct_data2):
    image_id_to_prob1 = get_image_id_to_prob(struct_data1)
    image_id_to_prob2 = get_image_id_to_prob(struct_data2)
    all_image_ids = [x for x in image_id_to_prob1.keys() if x in image_id_to_prob2]
    all_image_ids_dict = {x: True for x in all_image_ids}
    image_id_to_prob1 = {}
