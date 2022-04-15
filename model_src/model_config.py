class ModelConfig:
    """ Contains configuration settings for our clustering models.


    pretraining_method: The method for pre-training the underlying visual model, used to extract image embeddings

    classifier: The type of classifier used on top of the backbone model

    svm_kernel: The type of kernel used, in case an SVM classifier is used

    classifier_layer_size: The list of sizes of classifier layers, in case a neural classifier is used

    classifier_activation_func: The activation function, in case a neural classifier is used

    struct_property: The linguistic structural property that the model is trained to predict

    learning_rate: The learning rate for the visual encoder's training
    """

    def __init__(self,
                 pretraining_method='image_net',
                 classifier='neural',
                 svm_kernel='rbf',
                 classifier_layer_size=[],
                 classifier_activation_func='relu',
                 use_batch_norm=False,
                 struct_property='passive',
                 learning_rate=1e-4,
                 ):
        super(ModelConfig, self).__init__()

        self.pretraining_method = pretraining_method
        self.classifier = classifier
        self.svm_kernel = svm_kernel
        self.classifier_layer_size = classifier_layer_size
        self.classifier_activation_func = classifier_activation_func
        self.use_batch_norm = use_batch_norm
        self.struct_property = struct_property
        self.learning_rate = learning_rate

    def __str__(self):
        return 'Configuration: ' + str(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, ModelConfig):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__
