import numpy as np
import cv2
from Camera import Camera
import time
from abc import ABC
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import subprocess
import os
import signal




class IpCam(Camera, ABC):

    def __init__(self, ipCamPath='/Users/Merlin/Documents/GitHub/IPcam/',
                 excChromePath='/Users/Merlin/Downloads/chromedriver', url='http://localhost:1234/', roomid='111',
                 exposure=0.1, white_balance=0, auto_focus=True, grayscale=True):
        # Setting and initializing the external camera
        resolution = (4032, 3024)
        fps = 1
        # Init base class
        super().__init__(exposure, white_balance, auto_focus, fps, resolution, grayscale)
        self.ipCamPath = ipCamPath
        # Start server running bash script
        ipCamPathScript = ipCamPath + 'ios-ip-cam.sh ' + ipCamPath
        #rc = subprocess.call(ipCamPathScript, shell=True)
        self.pro = subprocess.Popen(ipCamPathScript, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        # Grab current frame
        self.driver = webdriver.Chrome(executable_path=excChromePath)
        # /Users/Merlin/Downloads/chromedriver
        #self.driver = webdriver.Chrome()
        # Go to your page url
        #url = 'http://google.com/'
        self.driver.get(url)
        # Enter room id
        username = self.driver.find_element_by_css_selector('input.MuiInputBase-input.MuiInput-input')
        username.send_keys(roomid)
        # Enter room
        enter = self.driver.find_element_by_css_selector('button.MuiButtonBase-root.MuiButton-root.MuiButton-outlined.makeStyles-button-3')
        enter.click()
        connect = self.driver.find_element_by_css_selector('button.MuiButtonBase-root.MuiButton-root.MuiButton-outlined.Video-button-184')
        connect.click()
        time.sleep(10)
        self.connected = True
        self.hdr_exposures = None

    def getFPS(self):
        # Returns the frame rate
        return self.fps

    def setFPS(self, fps):
        # Sets the on frame rate
        self.fps = fps

    def setAutoGain(self):
        raise NotImplementedError

    def getGain(self):
        raise NotImplementedError

    def setGain(self):
        raise NotImplementedError

    def getResolution(self):
        # Returns a tuple resolution (width, height)
        return self.resolution

    def setResolution(self, resolution):
        # Sets a new resolution
        # May only support certain resolutions
        self.resolution = resolution

    def getImage(self, name='test', saveImage=True, saveNumpy=True, calibration=False, timeout=5000):
        # Get button you are going to click by its id ( also you could us find_element_by_css_selector to get element by css selector)
        if self.connected:
            disconnect = self.driver.find_element_by_css_selector('button.MuiButtonBase-root.MuiButton-root.MuiButton-outlined.Video-button-184')
            disconnect.click()
            self.connected = False
        #take_photo = self.driver.find_element_by_css_selector('span.MuiButton-endIcon.MuiButton-iconSizeMedium')
        #take_photo.click()
        cap_num = name[-1]
        img_path = self.ipCamPath + 'ip-cam/results/' + cap_num + '.png'
        while not os.path.isfile(img_path):
            time.sleep(1)
        frame = cv2.imread(img_path)
        print(name, " :", frame.shape)
        if saveImage:
            if calibration:
                cv2.imwrite('CalibrationImages/' + name + '.PNG', frame)
            else:
                cv2.imwrite('CapturedImages/' + name + '.PNG', frame)
        if saveNumpy:
            if calibration:
                np.save('CalibrationNumpyData/' + name, frame)
            else:
                np.save('CapturedNumpyData/' + name, frame)
        return frame

    def setAutoExposure(self):
        raise NotImplementedError

    def setExposure(self, exposure):
        raise NotImplementedError

    def getExposure(self):
        raise NotImplementedError

    def viewCameraStream(self):
        raise NotImplementedError

    def quit_and_close(self):
        self.pro.kill()
        #os.killpg(os.getpgid(self.pro.pid), signal.SIGTERM)

    def quit_and_open(self):
        raise NotImplementedError

    def getStatus(self):
        raise NotImplementedError
