import torch
import torch.nn as nn

from model_src.image_linguistic_info_classifiers.image_linguistic_info_classifier import ImLingInfoClassifier


class InternalClassifier(nn.Module):
    def __init__(self, backbone_model_inference_func, backbone_output_size, layer_size_list, output_size,
                 freeze_backbone):
        super(InternalClassifier, self).__init__()

        self.freeze_backbone = freeze_backbone
        self.backbone_model_inference_func = backbone_model_inference_func

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        self.classification_head = self.get_classification_head(backbone_output_size, layer_size_list, output_size)
        self.classification_head.to(device)

    @staticmethod
    def get_classification_head(input_size, layer_size_list, class_num):
        layers = []
        cur_input_size = input_size
        for cur_output_size in layer_size_list:
            layers.append(nn.Linear(cur_input_size, cur_output_size))
            layers.append(nn.ReLU())
            cur_input_size = cur_output_size
        layers.append(nn.Linear(cur_input_size, class_num))
        layers.append(nn.ReLU())
        layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.backbone_model_inference_func(x)
        else:
            x = self.backbone_model_inference_func(x)

        return self.classification_head(x)


class ImLingInfoNeuralClassifier(ImLingInfoClassifier):

    """ This class predicts linguistic structural info of image descriptions, given only the images, using an MLP.
        It is based on a backbone visual model with a classification head for each predicted property.
    """

    def __init__(self, config, model_dir, model_name):
        super(ImLingInfoNeuralClassifier, self).__init__(config, model_dir, model_name)

        if self.config.pretraining_method == 'none':
            self.get_backbone_model().eval()
            self.get_backbone_model().requires_grad_(False)

        self.internal_classifier = InternalClassifier(
            self.backbone_model_inference, self.get_backbone_output_size(),
            self.config.classifier_layer_size, self.output_size, self.config.pretraining_method == 'none'
        )

    def inference(self, image_tensor):
        return self.internal_classifier(image_tensor)

    def predict(self, image_tensor):
        with torch.no_grad():
            output = self.inference(image_tensor)
        predictions = output.argmax(dim=1)
        return predictions
