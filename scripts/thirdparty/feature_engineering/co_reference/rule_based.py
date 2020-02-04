from feature_engineering.co_reference.models import CoReference


class MultiPassSieveCoReference(CoReference):
    @staticmethod
    def process(document, entities, window_size=2):
        """

        :param document:
        :param entities:
        :param window_size:
        :return: list of new entities from co-reference
        """
        ret = list()
        # Do something here
        return ret
