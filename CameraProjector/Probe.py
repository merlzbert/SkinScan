from abc import abstractmethod


class Probe:
    def __init__(self, power):
        self.power = power # in mW

    @abstractmethod
    def set_power(self, power):
        # To override
        pass

    @abstractmethod
    def get_power(self):
        # To override
        pass

    @abstractmethod
    def quit_and_close(self):
        # To override
        pass
