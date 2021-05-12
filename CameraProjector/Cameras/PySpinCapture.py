import math, time
import numpy as np
import cv2
from .Camera import Camera
from abc import ABC

import PySpin
# Install PySpin from the .whl in the Spinnaker sdk https://www.flir.com/products/spinnaker-sdk/
# click download and select the appropiate OS and Python version.

class PySpinCapture(ABC, Camera):
    @staticmethod
    def list_devices():
        """
        Returns the list of connected camera devices. 
        """
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        num_cameras = cam_list.GetSize()
        print ("There are", num_cameras, "cameras available")
        return cam_list

    def __init__(self, index=0, isMonochrome=False, is16bits=False, width= None, height= None):
        """
        Initializes the PySpinCamera class. 
        input
        index: the index of the camera to be used
        output
        """
        # [Current Support] Single camera usage(select by index)
        self._system = PySpin.System.GetInstance()
        # Get current library version
        self._version = self._system.GetLibraryVersion()
        print('Library version: {}.{}.{}.{}\n'.format(self._version.major, self._version.minor, self._version.type,
                                self._version.build))
        self.index = index
        self._cameraList = self.list_devices()
        
        if self._cameraList.GetSize() >= self.index:
            raise Exception(f"Can't load camera {index}. Camera index exceeds number of cameras.")

        self._camera = self._cameraList.GetByIndex(index)
        self._camera.Init()
        self._isMonochrome = isMonochrome
        self._is16bits = is16bits

        self._nodemap = self._camera.GetNodeMap()
        self._init_nodes()

        self.width = width or self._node_width.GetMax()
        self.height = height or self._node_height.GetMax()

        self.setAcquisitMode(1)
        self.setPixel()
        self.setResolution(self.width, self.height)
        self.setCamAutoProperty(False)

        #Print device info
        self.camera_model = self._node_device_name.GetValue()
        self.device_serial_number = self._node_device_serial_number.GetValue()
        print("Initializing", self.camera_model)
        print("Serial number: ", self.device_serial_number)

        # Filling in values for abstract class initialization
        # update if needed. 
        exposure = self.getExposure()
        fps = self.getFPS()
        white_balance = 0 # random fill value for required field.
        auto_focus = True # random fill value for required field. 

        # self._camera.PixelFormat.GetValue() try this
        resolution = self.getResolution()
        grayscale = isMonochrome
        super().__init__(exposure, white_balance, auto_focus, fps, resolution, grayscale)

    def _init_nodes(self):
        """
        Initializes the nodes of the Spinnaker sdk. Each node has a GetValue and a SetValue attribute. 
        """
        # model info nodes
        self._node_device_name = self._get_node('DeviceModelName', "string")
        self._node_device_serial_number = self._get_node("DeviceSerialNumber", "string")
        # Acquisition mode nodes
        self._node_acquisition_mode = self._get_node("AcquisitionMode", node_type= "enumeration", 
                                                    check_writable= True)
        self._node_acquisition_frame_rate = self._get_node("AcquisitionFrameRate", "float", check_writable=True)
        # Image size nodes
        self._node_width = self._get_node("Width", "integer")
        self._node_height = self._get_node("Height", "integer")
        # Exposure nodes
        self._node_exposure_time = self._get_node("ExposureTime", "float")
        self._node_exposure_auto = self._get_node("ExposureAuto", "enumeration")
        # Gain nodes
        self._node_gain_auto = self._get_node("GainAuto", "enumeration", check_writable= True)
        self._node_gain = self._get_node("Gain", "float")
        # Gamma node
        self._node_gamma_enable = self._get_node("GammaEnable", "boolean",check_writable= True)
        # Pixel format nodes
        self._node_pixel_format = self._get_node("PixelFormat", "enumeration")
        # legacy init for other parameters.  
        self._attribute_init()

    def _get_node(self, node_name, node_type= "enumeration", check_writable= False):
        """
        Loads a single Spinnaker node
        input:
        node_name: the node name given by Spinnaker to the desired node, input for the _nodemap.GetNode() function
        node_type: the type of C pointer for the desired node. Options include:
            - enumeration
            - integer
            - float
            - string
            - boolean
        check_writable: check if node isAvailable and isWritable before returning, and raise exception if it is not. 
        output: 
        node: Spinnaker node with GetValue and SetValue options. 
        """
        node_type= str.lower(node_type)

        if node_type == "enumeration":
            current_node = PySpin.CEnumerationPtr(self._nodemap.GetNode(node_name))
        elif node_type == "integer":
            current_node = PySpin.CIntegerPtr(self._nodemap.GetNode(node_name))
        elif node_type == "float":
            current_node = PySpin.CFloatPtr(self._nodemap.GetNode(node_name))
        elif node_type == "string":
            current_node = PySpin.CStringPtr(self._nodemap.GetNode(node_name))
        elif node_type == "boolean":
            current_node = PySpin.CBooleanPtr(self._nodemap.GetNode(node_name))
        else:
            raise Exception(f"unknown node type {node_type}")

        if check_writable:
            if not PySpin.IsAvailable(current_node) or not PySpin.IsWritable(current_node):
                self.print_retrieve_node_failure('node', node_name)
                raise Exception(f"Error loading node. Node {node_name} is not writeable (see stdout print output)")
        
        return current_node

    def _attribute_init(self):
        # Retrieve entry node from enumeration node
        nodeContinuousAcquisition = self._node_acquisition_mode.GetEntryByName("Continuous")
        if not PySpin.IsAvailable(nodeContinuousAcquisition) or not PySpin.IsReadable(
                nodeContinuousAcquisition):
            print("Unable to set acquisition mode to continuous (entry retrieval). Aborting...")
            return False

        self.nodeAcquisitionContinuous = nodeContinuousAcquisition.GetValue()

        # Retrieve entry node from enumeration node
        nodeSingleAcquisition = self._node_acquisition_mode.GetEntryByName("SingleFrame")
        if not PySpin.IsAvailable(nodeSingleAcquisition) or not PySpin.IsReadable(
                nodeSingleAcquisition):
            print("Unable to set acquisition mode to Single Frame (entry retrieval). Aborting...")
            return False
        self.nodeAcquisitionSingle = nodeSingleAcquisition.GetValue()

        # Pixel Format Node
        nodePixelFormatMono8 = PySpin.CEnumEntryPtr(
            self._node_pixel_format.GetEntryByName('Mono8'))
        self.pixelFormatMono8 = nodePixelFormatMono8.GetValue()
        nodePixelFormatMono16 = PySpin.CEnumEntryPtr(
            self._node_pixel_format.GetEntryByName('Mono16'))
        self.pixelFormatMono16 = nodePixelFormatMono16.GetValue()

        # nodePixelFormatRGB8 = PySpin.CEnumEntryPtr(
        #     self._node_pixel_format.GetEntryByName('BayerRG8'))
        # self.pixelFormatRGB8 = nodePixelFormatRGB8.GetValue()
        #
        # nodePixelFormatRGB16 = PySpin.CEnumEntryPtr(
        #     self._node_pixel_format.GetEntryByName('BayerRG16'))
        # self.pixelFormatRGB16 = nodePixelFormatRGB16.GetValue()

        # Exposure Node
        self._exposureAutoOff = self._node_exposure_auto.GetEntryByName('Off')
        if not PySpin.IsAvailable(self._exposureAutoOff) or not PySpin.IsReadable(self._exposureAutoOff):
            self.print_retrieve_node_failure('entry', 'ExposureAuto Off')
            return False

        self._exposureAutoOn = self._node_exposure_auto.GetEntryByName('Continuous')
        self._exposureMin = self._node_exposure_time.GetMin()
        self._exposureMax = self._node_exposure_time.GetMax()

        self._gainAutoOff = self._node_gain_auto.GetEntryByName('Off')
        if not PySpin.IsAvailable(self._gainAutoOff) or not PySpin.IsReadable(self._gainAutoOff):
            self.print_retrieve_node_failure('entry', 'GainAuto Off')
            return False

        self._gainAutoOn = self._node_gain_auto.GetEntryByName('Continuous')
        self._gainMin = self._node_gain.GetMin()
        self._gainMax = self._node_gain.GetMax()
        
        self.setFPS(5)

    """
    Abstract camera class methods
    """

    def getImage(self):
        """
        Captures an image and returns a corresponding numpy array. 
        input:
        none
        output:
        ret: True if successful or False if error. 
        frame: the numpy array representation of the image. 
        """
        self.setAcquisitMode(0)
        self._camera.BeginAcquisition()
        cameraBuff = self._camera.GetNextImage()
        if cameraBuff.IsIncomplete():
            return False, None

        cameraImg = cameraBuff.GetData().reshape(self.height, self.width)
        image = cameraImg.copy()
        cameraBuff.Release()
        self._camera.EndAcquisition()

        if self._isMonochrome:
            return True, image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
            return True, image_rgb

    def getExposure(self):
         # Get the value of exposure time.
        if self._camera.ExposureTime.GetAccessMode() == PySpin.RW or self._camera.ExposureTime.GetAccessMode() == PySpin.RO:
            # The exposure time is retrieved in µs so it needs to be converted to ms to keep consistency.
            exposure = (int)(self._camera.ExposureTime.GetValue() / 1000)
            return exposure
        else:
            print ('Unable to get exposure time. Aborting...')
            return False

    def setExposure(self, exposure_to_set):

        exposure_to_set = float(exposure_to_set)
        if exposure_to_set < self._exposureMin or exposure_to_set > self._exposureMax:
            print("[WARNING]: Gain value should be within {} to {}.(Input:{}) Set to half."
                  .format(self._exposureMin, self._exposureMax, exposure_to_set))
            self._node_exposure_time.SetValue(math.floor(self._exposureMax + self._exposureMin))
            print("Exposure: {}".format(self._node_exposure_time.GetValue()))
        else:
            self._node_exposure_time.SetValue(exposure_to_set)
            print("Exposure: {}".format(self._node_exposure_time.GetValue()))

    def getFPS(self):
        # get the value of fps
        fps = self._node_acquisition_frame_rate.GetValue()
        return fps

    def setFPS(self, fps):
        self._node_acquisition_frame_rate.SetValue(fps)

    def getGainAuto(self):
        print("please use getAutoGain instead of getGainAuto")
        return self.getAutoGain()

    def getAutoGain(self):
        gain_status = self._node_gain_auto.GetValue()
        return gain_status

    def setGainAuto(self, *args, **kwargs):
        print ("please use setAutoGain")
        return self.setAutoGain(*args, **kwargs)

    def setAutoGain(self, gain_status= 'off'):
        gain_status = str.title(gain_status)
        if gain_status not in ["On", "Off"]:
            raise Exception("Status must be on or off")

        gain_auto_status_code = self._node_gain_auto.GetEntryByName(gain_status)
        self._node_gain_auto.SetIntValue(gain_auto_status_code.GetValue())
        return

    def getGain(self):
        return self._node_gain.GetValue()

    def setGain(self, gain_to_set):

        if float(gain_to_set) < self._gainMin or float(gain_to_set) > self._gainMax:
            print("[WARNING]: Gain value should be within {} to {}.(Input:{}) Set to half."
                  .format(self._gainMin, self._gainMax, float(gain_to_set)))
            self._node_gain.SetValue(math.floor(self._gainMax + self._gainMin))
            print("Gain: {}".format(self._node_gain.GetValue()))
        else:
            self._node_gain.SetValue(float(gain_to_set))
            print("Gain: {}".format(self._node_gain.GetValue()))

    def getResolution(self):
        width = self._node_width.GetValue()
        height = self._node_height.GetValue()
        return (width, height)

    def setResolution(self, width, height):
        self._node_width.SetValue(width)
        self._node_height.SetValue(height)

    def viewCameraStream(self):
        raise NotImplementedError
    
    def quit_and_close(self):
        return self.release()

    def quit_and_open(self):
        return self.reset()

    def getStatus(self):
        # to implement, search for camera status node in spinnaker sdk. 
        return None

    """
    Other class methods
    """

    def setAcquisitMode(self, mode=0):
        if mode==0:
            #Single Frame mode
            self._node_acquisition_mode.SetIntValue(self.nodeAcquisitionSingle)
        elif mode==1:
            # Continuous mode
            self._node_acquisition_mode.SetIntValue(self.nodeAcquisitionContinuous)
            #self._node_acquisition_mode.SetValue(PySpin.AcquisitionMode_Continuous)

        print(self._node_acquisition_mode.GetIntValue())

    def setPixel(self):
        # Set the pixel format.
        if self._isMonochrome:
            # Enable Mono8 mode.
            if self._is16bits:
                self._node_pixel_format.SetIntValue(self.pixelFormatMono16)
            else:
                self._node_pixel_format.SetIntValue(self.pixelFormatMono8)
        # else:
        #     # Enable RGB8 mode.
        #     if self._is16bits:
        #         self._node_pixel_format.SetIntValue(self.pixelFormatRGB16)
        #     else:
        #         self._node_pixel_format.SetIntValue(self.pixelFormatRGB8)


    def setCamAutoProperty(self, switch=True):
        # [Current Support] Gain, Exposure time
        # In order to manual set value, turn off auto first
        if switch:
            if PySpin.IsAvailable(self._exposureAutoOn) and PySpin.IsReadable(self._exposureAutoOn):
                self._node_exposure_auto.SetIntValue(self._exposureAutoOn.GetValue())
                print('Turning automatic exposure back on...')

            if PySpin.IsAvailable(self._gainAutoOn) and PySpin.IsReadable(self._gainAutoOn):
                self._node_gain_auto.SetIntValue(self._gainAutoOn.GetValue())
                print('Turning automatic gain mode back on...\n')

            self._node_gamma_enable.SetValue(True)
        else:
            self._node_exposure_auto.SetIntValue(self._exposureAutoOff.GetValue())
            print('Automatic exposure disabled...')

            self._node_gain_auto.SetIntValue(self._gainAutoOff.GetValue())
            print('Automatic gain disabled...')

            #self._node_gamma_enable.SetValue(False)
            #print('Gamma disabled...')

    def grabFrame(self, *args, **kwargs):
        return self.getImage(*args, **kwargs)

    def grabFrameCont(self):
        cameraBuff = self._camera.GetNextImage()
        if cameraBuff.IsIncomplete():
            return False, None

        cameraImg = cameraBuff.GetData().reshape(self.height, self.width)
        image = cameraImg.copy()
        cameraBuff.Release()
        #self._camera.EndAcquisition()

        if self._isMonochrome:
            return True, image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
            return True, image_rgb

    def beginAcquisit(self, switch=True):
        if switch:
            self._camera.BeginAcquisition()
        else:
            self._camera.EndAcquisition()

    def getNextImage(self):
        return self._camera.GetNextImage()

    def setupHDR(self):

        if PySpin.IsAvailable(self._exposureAutoOn) and PySpin.IsReadable(self._exposureAutoOn):
            self._node_exposure_auto.SetIntValue(self._exposureAutoOn.GetValue())
            print('Turning automatic exposure back on...')

        time.sleep(0.5)
        series = [2 ** (-2), 2 ** (-1), 1, 2 ** 1, 2 ** 2]
        midExposure = self._node_exposure_time.GetValue()
        print("midExposure: ", midExposure)
        self.exposureHDRList = [midExposure * x for x in series]
        if self.exposureHDRList[0] < self._exposureMin:
            self.exposureHDRList[0] = self._exposureMin
        if self.exposureHDRList[-1] > self._exposureMax:
            self.exposureHDRList[-1] = self._exposureMax

        print("HDR Exposure List: ", self.exposureHDRList)

        self._node_exposure_auto.SetIntValue(self._exposureAutoOff.GetValue())
        print('Automatic exposure disabled...')

    def captureHDR(self):
        if not hasattr(self, 'exposureHDRList'):
            print("[ERROR]: Need to setup HDR Exposure list first!!!")
            return 0
        imgs = np.zeros((len(self.exposureHDRList), self.height, self.width))
        for index, x in enumerate(self.exposureHDRList):
            self.setExposure(x)
            flag, tmp = self.grabFrame()
            if flag:
                imgs[index, ...] = tmp
            else:
                print("[WARNING]: Invalid Capture!!!")

        return imgs

    def release(self):

        # Turn auto gain and exposure back on in order to return the camera to tis default state
        self.setCamAutoProperty(True)
        if self._camera.IsStreaming():
            self._camera.EndAcquisition()
        self._camera.DeInit()
        #del self._camera
        # self._cameraList.Clear()
        # self._system.ReleaseInstance()

    def __enter__(self):
        return self

    def reset(self):
        self.__init__()

    def __exit__(self):
        self.release()

    def print_retrieve_node_failure(self, node, name):

        print('Unable to get {} ({} {} retrieval failed.)'.format(node, name, node))
        print('The {} may not be available on all camera models...'.format(node))
        print('Please try a Blackfly S camera.')


######
# CLASS FOR COMPATIBILITY WITH NEURAL-HOLOGRAPHY REPOSITORY.
######


class CameraCapture:
    """
    A class compatible with the neural-holography repo that holds more than 1 camera. 
    """
    def __init__(self):
        # [Current Support] Single camera usage(select by index)
        self._system = PySpin.System.GetInstance()
        # Get current library version
        self._version = self._system.GetLibraryVersion()
        print('Library version: {}.{}.{}.{}\n'.format(self._version.major, self._version.minor, self._version.type,
                                self._version.build))

        self.camera_count = len(self._system.GetCameras())
        print(self.camera_count, "cameras available.")

        # create an list of all devices connected and initialize them. 
        self._camera_list = [PySpinCapture(index= i, isMonochrome=False, is16bits=False) for i in range(self.camera_count)]

        self._current_camera = 0
    
    @property
    def current_camera(self):
        """Camera to which the 'CameraCapture' class is currently connected"""
        return self._current_camera

    @current_camera.setter
    def current_camera(self, camera_idx):
        print("Changing current camera to", camera_idx)
        self._current_camera = camera_idx


    def connect(self, camera_idx):
        self.current_camera = camera_idx
        return

    def disconnect(self):
        # return self.camera_list[self.current_camera].release()
        return
    
    def toggle_embedded_timestamp(self, enable_timestamp):
        # Not sure that there's a corresponding method for PySpin or what the purpose is
        pass

    def grab_images(self, num_images_to_grab=1):
        image_list = []
        # grab the requested number of images
        for i in range(num_images_to_grab):
            camera = self._camera_list[self.current_camera]
            success, image_rgb = camera.grabFrame()
            image_list.append(image_rgb)
            if success == False:
                print("Error capturing image")
        
        return image_list

    def start_capture():
        # Not sure that there's a corresponding method for PySpin or if it is needed
        pass

    def stop_capture():
        # Not sure that there's a corresponding method for PySpin or if it is needed
        pass



