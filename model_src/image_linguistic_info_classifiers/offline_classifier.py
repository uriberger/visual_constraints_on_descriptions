import torch

from model_src.image_linguistic_info_classifiers.image_linguistic_info_classifier import ImLingInfoClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class ImLingInfoOfflineClassifier(ImLingInfoClassifier):

    """ This class predicts linguistic structural info of image descriptions, given only the images, using SVM.
        It is based on a backbone visual model with an offline (svm/random forest) classifier head for each predicted
        property.
    """

    def __init__(self, config, model_dir, model_name):
        super(ImLingInfoOfflineClassifier, self).__init__(config, model_dir, model_name)

        if config.pretraining_method == 'none':
            self.get_backbone_model().eval()

        if config.classifier == 'svm':
            self.clf = SVC(kernel=config.svm_kernel)
        elif config.classifier == 'random_forest':
            self.clf = RandomForestClassifier()

    def fit(self, training_mat, label_mat):
        self.clf.fit(training_mat, label_mat)

    def predict(self, image_tensor):
        with torch.no_grad():
            extracted_features = self.backbone_model_inference(image_tensor).cpu()
        return torch.from_numpy(self.clf.predict(extracted_features))