class Datastore:
    def get_data(self):
        """
        This will return the Training Data subset
        """
        pass

    # def get_pre_train_data(self):
    #     """
    #     This will return the Pre-training Data Subset
    #     """
    #     pass

    def get_testing_data(self):
        """
        This will return the Testing Data subset
        """
        pass


class CombinedDatastore(Datastore):

    def __init__(self, x_train, y_train, x_target, y_target) -> None:
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_target = x_target
        self.y_target = y_target

    def get_data(self):
        return (self.x_train, self.y_train, None), (None, None, None)

    def get_testing_data(self):
        return self.x_target, self.y_target, None
