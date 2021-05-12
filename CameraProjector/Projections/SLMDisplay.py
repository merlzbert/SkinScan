import ctypes
import numpy as np
from .slm_utils import _slm_win as slm
from PIL import Image
from pathlib import Path

from .Projection import Projection
from abc import ABC

class NoSLMError(Exception):
    pass

class SLMDisplay(ABC, Projection):
    """
    Class to handle SLM input-ouput. 
    
    Methods:
    list_DVI (static method): obtain list of available displays connected with DVI
    __init__: create display object with device number, mode, width and height
    open: initialize display
    close: clean resources
    display: display image from numpy array, csv file or image file
    configure: configure wavelength and phase of the display
    
    Attributes:
    status: returns the slm status. 
    width: returns the display width
    height: returns the display height
    """

    def __init__(self, display_index= 0, display_mode= "USB", width= 1920, height= 1200, verbose= True):
        """
        Create SLMDisplay object. Parameters:
        display_index: the index of the SLM display to be initialized
        display_mode: either "DVI" or "USB".
        width: width of the display (1920 by default)
        height: height of the display (1080 by default)
        verbose: to enable printing of statements for debugging. 

        Can have more than one display at one. Example with 2 displays
        with SLMDisplay(params) as d1, SLMDisplay(params2) as d2: 
            write code here. use d1.method() or d2.method() to interact with the displays
        """
        self.super().__init__(self, (width, height), None)

        self.idx = display_index

        # set "DVI" or "USB" mode. DVI mode is 1, USB mode is 0.
        if display_mode.upper() == "DVI":
            self.mode = 1 
        elif display_mode.upper() == "USB":
            self.mode = 0
        else: 
            raise Exception(display_mode + " is not a valid display mode. Use 'DVI' or 'USB'")

        self.mode_string = display_mode.upper()
        self.width = width
        self.height = height
        self.size = (self.width, self.height)

        self.vprint = print if verbose else lambda *x, **y: None



        with self:
            # Obtain status
            self._status = slm.SLM_Ctrl_ReadSU(self.idx)

            # Obtain current wavelength and phase
            p_wavelength = ctypes.c_uint32(0)
            p_phase = ctypes.c_uint32(0)
            ret = slm.SLM_Ctrl_ReadWL(self.idx, p_wavelength, p_phase)
            if ret != 0:
                self.vprint(f"Display {self.idx} has no phase or wavelength information")
                self._wavelength = self._phase = None
            else:
                self._wavelength = p_wavelength.value
                self._phase = p_phase.value
                self.vprint(f"Current wavelength: {self.wavelength}\n"
                            f"Current phase: {self.phase}")

    @property
    def wavelength(self):
        """Wavelength of the SLM. Use display.configure() to assign a new value. """
        return self._wavelength

    @property
    def phase(self):
        """Phase of the SLM. Use display.configure() to assign a new value.""" 
        return self._phase

    @property
    def status(self):
        """Returns the current status of the SLM. Cannot be assigned a value.""" 
        self._status = slm.SLM_Ctrl_ReadSU(self.idx)
        return self._status

    def configure(self, wavelength= None, phase= None):
        """
        Set the wavelength and the phase of the SLM. Assume same values by default. 
        NOTE: valid inputs are 450-1600 for wavelength, 0-999 for phase
        """
        # Raise exception for invalid input
        if wavelength is not None:
            if wavelength < 450 or wavelength > 1600:
                raise Exception("Detected invalid input for wavelength not in the range of 450-1600")
        if phase is not None:
            if phase < 0 or phase > 999:
                raise Exception("Detected invalid input for phase not in the range of 0-999")

        # If user does not supply values, assume same values
        wavelength = int(wavelength) if wavelength else self._wavelength
        phase = phase if phase else self._phase
        self.vprint(f"Current wavelength: {self._wavelength}\tTarget wavelength: {wavelength}")
        self.vprint(f"Current phase: {self._phase}\t\tTarget phase: {phase}")

        # Make sure phase and wavelength are defined before proceeding, otherwise, raise exception
        if not phase or not wavelength:
            raise Exception(f"Please define wavelength and phase to configure! Target phase is {phase} and target "
            f"wavelength is {wavelength}.")

        # Make the change to the display and save it. 
        with self:
            ret = slm.SLM_Ctrl_WriteWL(self.idx, wavelength, phase)
            ret2 = slm.SLM_Ctrl_WriteAW(self.idx)
            if ret == 0 and ret2 == 0:
                self._wavelength = wavelength
                self._phase = phase
            else:
                if ret == -200 or ret2 == -200:
                    raise NoSLMError(f"Could not configure SLM. Error -200, no SLM connected")
                else:
                    raise Exception (f"Could not configure SLM. Error code {ret} for setting values"
                                    f" and Error code {ret2} for saving the new values.")

        return ret
            


    def __enter__(self):
        return self.open()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        return self.close()

    def open(self):
        """
        Initialize display. Returns slm_status of 0, if there are no errors or a SLM error code.
        """
        # Open SLM
        self.vprint(f"Opening display {self.idx}")
        if self.mode_string == "DVI":
            slm.SLM_Disp_Open(self.idx)
        elif self.mode_string == "USB":
            slm.SLM_Ctrl_Open(self.idx)

        # Set mode, and recover error code as slm_status
        slm_status = slm.SLM_Ctrl_WriteVI(self.idx, self.mode)
        mode_string = "DVI" if self.mode else "USB"
        self.vprint(f"Setting mode to {mode_string} with SLM status: {slm_status}")
        return self

    def close(self):
        """
        Close display.
        """
        self.vprint(f"Closing display {self.idx}")
        if self.mode_string == "DVI":
            slm.SLM_Disp_Close(self.idx)
        elif self.mode_string == "USB":
            slm.SLM_Ctrl_Close(self.idx)
            
    def display(self, source, memory_number= 1):
        """
        Displays np.array, csv or png files.
        """
        # If it is a filepath, call _display_csv or _display_img
        if isinstance(source, (str, Path)):
            filepath = Path(source)
            filepath = filepath.resolve()
            file_suffix = filepath.suffix
            if not filepath.exists():
                raise Exception ("Invalid display source, enter an array or a filepath to a csv or png file.")
            else:
                filepath = str(filepath)
                if file_suffix == ".csv":
                    return self._display_csv(filepath)
                else:
                    try:
                        return self._display_img(filepath)
                    except:
                        raise
        #if it is an array, call _display_numpy
        elif isinstance(source, np.ndarray):
            return self._display_numpy(source, memory_number = memory_number)

        else:
            raise Exception("The display source must be a filepath to a csv or image file or a numpy array\n"
            f"The source data is not valid {source}"
            )

    def _display_numpy(self, data: np.uint16, memory_number=1):
        """
        Displays a numpy array in the current mode. Parameters:
        data: np.array of dtype np.ushort. can be grayscale or RGB. 
        IMPORTANT: the width and height of the data must be resized to the width and height of the display
        """
        # convert RGB data to 10bit. 
        if data.ndim == 3:
            data = SLMDisplay.convert_RGB_10bit(data)

        # Check dtype and size
        if data.dtype != np.ushort:
            data = data.astype(np.ushort)
        if data.shape[0] != self.height or data.shape[1] != self.width:
            data = self._resize_to_display(data)

        # create c_data
        c_data = data.ctypes.data_as(ctypes.POINTER((ctypes.c_ushort * self.height) * self.width)).contents
        
        # Display in DVI or USB mode
        slm_status = None
        if self.mode_string == "DVI":
            slm_status = slm.SLM_Disp_Data(self.idx, self.width, self.height, 0, c_data)
        elif self.mode_string == "USB":
            # Cast to pointer if using USB mode
            c_data = ctypes.cast(c_data, ctypes.POINTER(ctypes.c_ushort))
            # Display greyscale to modify memory, otherwise, it is read only
            slm_status = slm.SLM_Ctrl_WriteGS(self.idx, 0)
            # Update memory with appropiate data
            slm_status = slm.SLM_Ctrl_WriteMI(self.idx, memory_number, self.width, self.height, 0, c_data)
            # Display memory again
            slm_status = slm.SLM_Ctrl_WriteDS(self.idx, memory_number)
            self.vprint("Displaying", data.dtype, "on display", self.idx, "with status", slm_status)

        if slm_status != 0:
            self.vprint(f"Error when displaying {data.dtype}, slm status: {slm_status}")
        pass
        
        return slm_status

    def _display_csv(self, filepath, memory_number= 1):
        """
        Displays a csv file in the current mode. Parameters:
        filepath 
        IMPORTANT: the width and height of the data must be resized to the width and height of the display
        """
        # Display on DVI or USB mode
        slm_status = None
        if self.mode_string == "DVI":
            slm_status = slm.SLM_Disp_ReadCSV(self.idx, 0, filepath)
        elif self.mode_string == "USB":
            # Display greyscale pattern to be able to modify memory
            slm_status = slm.SLM_Ctrl_WriteGS(self.idx, 0)
            # Modify memory
            slm_status = slm.SLM_Ctrl_WriteMI_CSV(self.idx, memory_number, 0, filepath)
            # Display from memory again
            slm_status = slm.SLM_Ctrl_WriteDS(self.idx, memory_number)
            self.vprint("Displaying", filepath, "on display", self.idx, "with status", slm_status)            

        return slm_status

    def _display_img(self, filepath, memory_number= 1):
        """
        Displays a image file in the current mode. Parameters:
        filepath 
        """
        #Load file and resize to match display shape
        image_data = np.array(Image.open(filepath).resize((self.width, self.height)))
        return self._display_numpy(image_data, memory_number= memory_number)
        
    def _resize_to_display(self, data):
        #specified resample= 0 to correct a bug of the PIL package. 
        new_data = Image.fromarray(data).resize((self.width, self.height), resample= 0)
        new_data = np.asarray(new_data, dtype= np.ushort)
        return new_data

    @staticmethod
    def list_DVI() -> dict:
        """
        Lists DVI connected displays (not USB!). 
        Prints a list of all the available displays connected to the computer by DVI, including SLM.
        Returns a list of dictionaries with attributes display_index, width, height, name for each display.
        """
        display_list = []
        #get up to 8 first displays connected to the computer
        for display_idx in range(8):
            p_W = ctypes.c_ushort(0) # initialize C pointer, same as c_uint16
            p_H = ctypes.c_uint16(0) # initialize C pointer, same as c_ushort
            p_name = ctypes.create_string_buffer(128) # initialize pointer to 128-byte buffer for DisplayName
            
            slm_status = slm.SLM_Disp_Info2(display_idx, p_W, p_H, p_name)
            
            p_name = bytes(p_name.value)

            try: #try to decode device name to string using unicode. 
                p_name = p_name.decode('mbcs')
            except:
                pass
                
            if slm_status != 0: # error reading display or no display found
                continue
            
            display_list.append(dict(display_index = display_idx, width = p_W.value, height = p_H.value, name = p_name))
            
            output_message = (
            f"Display number: {display_idx}\n"
            f"\tWidth: {p_W}\n"
            f"\tHeight: {p_H}\n"
            f"\tDisplay name: {p_name}\n")
            
            print(output_message)

        return display_list

    @staticmethod
    def convert_RGB_10bit(data: np.uint8) -> np.uint16:
        """
        Converts a 24-bit rgb image to 10-bit format. 
        input: image or array of images with 3 channels of type np.uint
        output: 10-bit encoded image or array of images
        """
        data = data.astype(np.uint16)
        
        red = data[...,0]
        green = data[...,1]
        blue = data[...,2]
        
        # convert 24-bit to 10-bit encoding
        # bits number 9, 8, 7 for red, 6, 5, 4 for green and 3, 2, 1, 0 for blue. 
        image_data_10bit = (red//32)*(2**7) + (green//32)*(2**4) + (blue//16)*(2**0)
        
        return image_data_10bit

    
    ######
    # METHODS FOR COMPATIBILITY WITH NEURAL-HOLOGRAPHY REPOSITORY.
    ######
    def connect(self):
        return self.open()
    
    def disconnect(self):
        return self.close()

    def show_data_from_file(self, filepath):
        error = self.display(filepath)
        assert(error == 0)
    
    def show_data_from_array(self, numpy_array):
        error = self.display(numpy_array)
        assert(error == 0)

    """
    methods for compatibility with projection abstract base class
    """

    def displayPatterns(self, patterns, camera):
        return self.display(patterns)
        
    def getResolution(self):
        return (self.width, self.height)

    def setResolution(self):
        raise NotImplementedError
    
    def quit_and_close(self):
        return self.close()
    
    def getStatus(self):
        return self.status

    


    





