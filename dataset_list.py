from collections import defaultdict


# Language, dataset, translated indicator
language_dataset_list = [
    ('English', ['COCO', 'flickr30', 'iaprtc12', 'pascal_sentences'], False),
    ('German', ['multi30k', 'iaprtc12'], False),
    ('Japanese', ['STAIR-captions', 'pascal_jp', 'YJCaptions'], False),
    ('Chinese', ['coco-cn', 'flickr8kcn', 'ai_challenger'], False),
    ('German', ['multi30k', 'de_coco'], True),
    ('French', ['multi30k'], True),
    ('Chinese', ['coco-cn'], True)
]

all_datasets = list(set([x for outer in [y[1] for y in language_dataset_list] for x in outer]))

translated_only_datasets = [x for x in all_datasets
                            if len([y for y in language_dataset_list if x in y[1] and (not y[2])]) == 0]


multilingual_dataset_name_to_original_dataset_name = {
    'multi30k': 'flickr30',
    'flickr8kcn': 'flickr30',
    'flickr30': 'flickr30',
    'STAIR-captions': 'COCO',
    'YJCaptions': 'COCO',
    'de_coco': 'COCO',
    'coco-cn': 'COCO',
    'COCO': 'COCO',
    'iaprtc12': 'iaprtc12',
    'pascal_jp': 'pascal_sentences',
    'pascal_sentences': 'pascal_sentences',
    'ai_challenger': 'ai_challenger'
}


def get_orig_dataset_to_configs():
    dataset_to_language_list = defaultdict(list)
    for language, dataset_list, translated in language_dataset_list:
        for dataset in dataset_list:
            dataset_to_language_list[dataset].append((language, translated))

    orig_to_multilingual_mapping = defaultdict(list)
    for multilingual_dataset_name, orig_dataset_name in multilingual_dataset_name_to_original_dataset_name.items():
        orig_to_multilingual_mapping[orig_dataset_name].append(multilingual_dataset_name)

    orig_dataset_to_configs = {}
    for orig_dataset_name, multilingual_dataset_list in orig_to_multilingual_mapping.items():
        based_datasets = multilingual_dataset_list
        configs = []
        for dataset_name in based_datasets:
            languages_list = dataset_to_language_list[dataset_name]
            configs += [(dataset_name, x[0], x[1]) for x in languages_list]

        orig_dataset_to_configs[orig_dataset_name] = configs

    return orig_dataset_to_configs
