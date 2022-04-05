# Language, dataset, translated indicator
language_dataset_list = [
    ('English', ['COCO', 'flickr30', 'iaprtc12', 'pascal_sentences'], False),
    ('German', ['multi30k', 'iaprtc12'], False),
    ('Japanese', ['STAIR-captions', 'pascal_jp', 'YJCaptions'], False),
    ('Chinese', ['coco-cn', 'flickr8kcn'], False),
    ('German', ['multi30k', 'de_coco'], True),
    ('French', ['multi30k'], True),
    ('Chinese', ['coco-cn'], True)
]

all_datasets = list(set([x for outer in [y[1] for y in language_dataset_list] for x in outer]))

translated_only_datasets = [x for x in all_datasets
                            if len([y for y in language_dataset_list if x in y[1] and (not y[2])]) == 0]
