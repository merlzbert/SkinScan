import sys
import glob
import os
import numpy as np
import cv2
from cv2 import aruco


class IntrinsicCalibration:
    def __init__(self, dict_aruco=cv2.aruco.DICT_4X4_250, sqWidth=11, sqHeight=8, checkerSquareSize=0.022,
                 markerSquareSize=0.016, criteria_eps=1e-9, criteria_count=10000):
        # Initialize ChArUco Board size values: default DINA4
        # Visit: https://calib.io/pages/camera-calibration-pattern-generator
        # To create: calib.io_charuco_279x215_8x11_24_DICT_4X4.pdf
        # Initialize dictionary, see more in OpenCV
        self.dict = dict_aruco
        # Amount of squares in width
        self.sqWidth = sqWidth
        # Amount of squares in heights
        self.sqHeight = sqHeight
        # Size of checker square on printed ChArUco board in meter
        self.checkerSquareSize = checkerSquareSize
        # Size of marker square on printed ChArUco board in meter
        self.markerSquareSize = markerSquareSize
        self.criteria_eps = criteria_eps
        self.criteria_count = criteria_count

    @staticmethod
    def readFileList(imgFolder, ImgPattern="*.PNG"):
        # Read all PNG files in folder
        imgFileList = glob.glob(os.path.join(imgFolder, ImgPattern))
        imgFileList.sort()
        return imgFileList

    def calibration(self, imgFolder='CalibrationImages/Intrinsic', distImgFolder='CalibrationImages/Distorted'):
        # Retrieve Images
        imgFileList = self.readFileList(imgFolder)
        # All Charuco Corners
        allCorners = []
        # All Charuco Ids
        allIds = []
        decimator = 0
        # Retrieve dictionary
        dictionary = aruco.getPredefinedDictionary(self.dict)
        board = cv2.aruco.CharucoBoard_create(self.sqWidth, self.sqHeight, self.checkerSquareSize,
                                              self.markerSquareSize, dictionary)
        # Loop through images
        for i in imgFileList:
            print("Reading %s" % i)
            # Load image to grayscale
            img = cv2.imread(i)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect markers
            [markerCorners, markerIds, rejectedImgPoints] = cv2.aruco.detectMarkers(gray, dictionary)
            # Draw markers
            if len(markerCorners) > 0:
                [ret, charucoCorners, charucoIds] = cv2.aruco.interpolateCornersCharuco(markerCorners, markerIds, gray,
                                                                                        board)
                if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 3:
                    allCorners.append(charucoCorners)
                    allIds.append(charucoIds)

                cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds, [0, 255, 0])
                cv2.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds, [0, 0, 255])

            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
            cv2.imshow('Frame', img)
            cv2.waitKey(0)  # any key
            decimator += 1
        print("NumImg:", len(allCorners))
        imsize = img.shape
        print(imsize)
        # Try Calibration
        try:
            # , aa, bb, viewErrors
            # Calibrate camera
            [ret, cameraMatrix, disCoeffs, rvecs, tvecs, _, _,
             perViewErrors] = cv2.aruco.calibrateCameraCharucoExtended(
                allCorners, allIds, board, (imsize[0], imsize[1]),
                None, None, flags=cv2.CALIB_RATIONAL_MODEL,
                criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, self.criteria_count, self.criteria_eps))
            # Print computed calibration results
            print("Rep Error:", ret)
            print("Camera Matrix:", cameraMatrix)
            print("Per View Errors:", perViewErrors)
            print("Distortion Coefficients:", disCoeffs)
            print("R vecs:", rvecs)
            # Save calibration results in dedicated folder for numpy data of calibration
            np.savez('CalibrationNumpyData/intrinsic_calibration.npz', ret=ret, mtx=cameraMatrix, dist=disCoeffs,
                     rvecs=rvecs, tvecs=tvecs)
            # Undistort the images
            imgDistortFolder = distImgFolder
            imgDistortFilelist = self.readFileList(imgDistortFolder)
            img_num = 0
            # Loop through images to undistort
            for j in imgDistortFilelist:
                imgDistort = cv2.imread(j)
                h = imgDistort.shape[0]
                w = imgDistort.shape[1]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, disCoeffs, (w, h), 1, (w, h))
                print('Image cropped', j)
                # Undistort
                dst = cv2.undistort(imgDistort, cameraMatrix, disCoeffs, None)
                print('Image undistorted', j)
                imgDistortFilename = os.path.join(imgDistortFolder, '/undistort', str(img_num) + '.png')
                cv2.imwrite(imgDistortFilename, dst)
                print('Image saved', j)
                img_num = img_num + 1

        except ValueError as e:
            print(e)
        except NameError as e:
            print(e)
        except AttributeError as e:
            print(e)
        except:
            print("calibrateCameraCharuco fail:", sys.exc_info()[0])
        # Close windows
        print("Press any key on window to exit")
        cv2.waitKey(0)  # any key
        cv2.destroyAllWindows()
