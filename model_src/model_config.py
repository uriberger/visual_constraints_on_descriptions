class ModelConfig:
    """ Contains configuration settings for our clustering models.


    pretraining_method: The method for pre-training the underlying visual model, used to extract image embeddings

    struct_property: The linguistic structural property that the model is trained to predict

    learning_rate: The learning rate for the visual encoder's training
    """

    def __init__(self,
                 pretraining_method='image_net',
                 classifier='neural',
                 struct_property='passive',
                 learning_rate=1e-4,
                 ):
        super(ModelConfig, self).__init__()

        self.pretraining_method = pretraining_method
        self.classifier = classifier
        self.struct_property = struct_property
        self.learning_rate = learning_rate

    def __str__(self):
        return 'Configuration: ' + str(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, ModelConfig):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__
