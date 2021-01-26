from abc import abstractmethod


class ImageProcessing:
    def __init__(self, capture_path):
        # Path to first dataset
        self.path = capture_path

    @abstractmethod
    def loadData(self):
        # To override: Load captured data
        pass

    @abstractmethod
    def computePhaseMaps(self):
        # To override: If applicable
        pass

    @abstractmethod
    def computeNormalMap(self, exposure):
        # To override: Compute normals
        pass

    @abstractmethod
    def computeDepthMap(self, exposure):
        # To override: Compute normals
        pass

    @abstractmethod
    def computePointCloud(self):
        # To override: Compute depth map - point cloud
        pass

    @abstractmethod
    def highPassFilter(self):
        # To override
        pass



