class Calibration:
    def __init__(self, radio_calib=None, intr_calib=None, geo_calib=None):
        # This class simply combines the calibration objects from the different calibration procedures
        self.radio_calib = radio_calib
        self.intr_calib = intr_calib
        self.geo_calib = geo_calib



