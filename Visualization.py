import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    def __init__(self, image_processing):
        # Initialize image processing object that contains normals, phase maps and others
        self.ip = image_processing

    def showAlbedo(self):
        # Show first frame as reference
        if len(self.ip.albedo.shape) >= 3:
            plt.imshow(self.ip.albedo.astype(np.uint8))
        else:
            plt.imshow(self.ip.albedo, cmap='gray')
        plt.title("Grayscale Reference Frame")
        plt.xlabel("Pixel location x")
        plt.ylabel("Pixel location y")
        plt.show()

    def showPhaseMaps(self):
        # Show phase maps in x and y direction
        n_phase_x = self.ip.diff_x
        n_phase_y = self.ip.diff_y
        # n_phase_x /= n_phase_x.max()
        # n_phase_y /= n_phase_y.max()
        f, ax = plt.subplots(2, 1)
        ax[0].imshow(n_phase_x)
        ax[0].set_title("Phase X")
        ax[0].set_xlabel("Pixel location x")
        ax[0].set_ylabel("Pixel location y")
        ax[1].imshow(n_phase_y)
        ax[1].set_title("Phase Y")
        ax[1].set_xlabel("Pixel location x")
        ax[1].set_ylabel("Pixel location y")
        plt.show()
        
    def showAllImages(self):
        # Show all captured images
        if len(self.ip.albedo.shape) >= 3:
            f, ax = plt.subplots(self.ip.frames_x.shape[3], self.ip.frames_y.shape[3]+1)
            for i in range(self.ip.frames_x.shape[3]):
                ax[i, 0].imshow(self.ip.frames_x[..., i])
                ax[i, 0].set_title("Capture X" + str(i))
                ax[i, 0].set_xlabel("Pixel location x")
                ax[i, 0].set_ylabel("Pixel location y")
                ax[i, 1].imshow(self.ip.frames_y[..., i])
                ax[i, 1].set_title("Capture Y" + str(i))
                ax[i, 1].set_xlabel("Pixel location x")
                ax[i, 1].set_ylabel("Pixel location y")
            ax[0, self.ip.frames_y.shape[3]].imshow(self.ip.frame_reflectivity)
            ax[0, self.ip.frames_y.shape[3]].set_title("Full illumination")
            ax[0, self.ip.frames_y.shape[3]].set_xlabel("Pixel location x")
            ax[0, self.ip.frames_y.shape[3]].set_ylabel("Pixel location y")
            ax[1, self.ip.frames_y.shape[3]].imshow(self.ip.albedo)
            ax[1, self.ip.frames_y.shape[3]].set_title("Mean image")
            ax[1, self.ip.frames_y.shape[3]].set_xlabel("Pixel location x")
            ax[1, self.ip.frames_y.shape[3]].set_ylabel("Pixel location y")
        else:
            f, ax = plt.subplots(self.ip.frames_x.shape[2], self.ip.frames_y.shape[2]+1)
            for i in range(self.ip.frames_x.shape[2]):
                ax[i, 0].imshow(self.ip.frames_x[:, :, i], cmap='gray')
                ax[i, 0].set_title("Phase shift X")
                ax[i, 0].set_xlabel("Pixel location x")
                ax[i, 0].set_ylabel("Pixel location y")
                ax[i, 1].imshow(self.ip.frames_y[:, :, i], cmap='gray')
                ax[i, 1].set_title("Phase shift Y")
                ax[i, 1].set_xlabel("Pixel location x")
                ax[i, 1].set_ylabel("Pixel location y")
            ax[0, self.ip.frames_y.shape[2]].imshow(self.ip.frame_reflectivity, cmap='gray')
            ax[0, self.ip.frames_y.shape[2]].set_title("Full illumination")
            ax[0, self.ip.frames_y.shape[2]].set_xlabel("Pixel location x")
            ax[0, self.ip.frames_y.shape[2]].set_ylabel("Pixel location y")
            ax[1, self.ip.frames_y.shape[2]].imshow(self.ip.albedo, cmap='gray')
            ax[1, self.ip.frames_y.shape[2]].set_title("Mean image")
            ax[1, self.ip.frames_y.shape[2]].set_xlabel("Pixel location x")
            ax[1, self.ip.frames_y.shape[2]].set_ylabel("Pixel location y")
        plt.show()

    def showNormals(self):
        # Show normals as RGB representation
        showN = np.array((self.ip.normals + 1) / 2.0 * 255).astype(int)
        plt.imshow(showN)
        plt.title("Normals")
        plt.xlabel("Pixel location x")
        plt.ylabel("Pixel location y")
        plt.show()

    def showQuiverNormals(self, stride=50):
        # Plot normals as arrows at strided positions
        crop_n_x = self.ip.normals[::stride, ::stride, 0]
        crop_n_y = self.ip.normals[::stride, ::stride, 1]
        # Create a mesh-grid of the image
        x = np.linspace(0, self.ip.normals.shape[1], crop_n_x.shape[1])
        y = np.linspace(0, self.ip.normals.shape[0],  crop_n_x.shape[0])
        X, Y = np.meshgrid(x, y)
        # Plot the normals shown on top of the albedo image
        plt.quiver(X, Y, crop_n_x, crop_n_y, color='g', scale=10)
        plt.imshow(self.ip.albedo, cmap='gray')
        plt.show()
