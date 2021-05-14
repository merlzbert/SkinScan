import sys
from serial.tools import list_ports
# !pip install pyserial

from .Probe import Probe
from abc import ABC

class Laser(ABC, Probe):
    @staticmethod
    def list_lasers():
        """
        Returns a list of all of the lasers available.
        """
        devices_found = list_ports.comports()
        for device in devices_found:
            print( "{} : ({}, {}, {})".format(i, device.device, device.description, device.hwid ) )
            i = i + 1
        return devices_found
    
    def __init__(self, laser_idx, baud= 112500, timeout= 1, verbose= True):
        """
        Instantiates a laser. The defaults are fine for a single laser. 
        """
        super().__init__(None)
        self.print = print if verbose else None
        self.print("Starting laser_idx", laser_idx)
        devices= Laser.list_lasers()
        self.laser = serial.Serial(devices[laser_idx].device, baud= baud, timeout= timeout)
        self._status = self._send_command("l?") #0 if OFF, 1 if ON
        self._target_power = self._send_command("p?")
        self._power = self._send_command("pa?")
        
        self.print("Started laser with serial number", self.get_serial())
        status_string = "ON" if self.status == 1 else "OFF"
        self.print("Laser is currently", status_string)
        self.print(f"Power target is set to: {self.target_power/1000:.2f} mW")
        self.print(f"Power is currently: {self.power/1000:.2f} mW")

    @property
    def status(self):
        """
        Returns ON or OFF laser status
        """
        self._status = self._send_command("l?")
        return self._status

    @property
    def target_power(self):
        """
        Returns set power (not actual power)
        """
        self._target_power = self._send_command("p?")
        return self._target_power

    @property
    def power(self):
        self._power = self._send_command("pa?")
        return self._power

    def _send_command(self, command, verbose= True):
        """
        Sends a command to a laser and returns the result
        """
        if command is not str:
            raise Exception("Invalid command for laser, not a string")

        termination = "\r\n" #always end \r \n
        if verbose:
            print("~Sent laser:", command)
        self.laser.write((command + termination).encode('ascii') ) # encode('ascii')
        result = self.laser.readline().decode('ascii')
        if verbose:
            print("Laser returned:", result)
        return result

    def set_power(self, target):
        """
        Sets specified target power (specified in mW)
        """
        power_watts = target/1000
        command = str("p " + power_watts)
        self.print("Setting power to", target, "mW")
        self._send_command(command)
        self._target_power = self._send_command("p?")
        return
        
    def open(self):
        """
        Prepares the laser to be used
        """
        # clean laser
        result = self._send_command("cf")
        # start with autostart (warmup)
        result = self._send_command("@cob1")
        # set target power to 0
        result = self._send_command("p 0.0")
        return self

    def close(self):
        """
        Cleans resources and closes laser
        """
        # clean laser
        result = self._send_command("cf")
        # close laser
        result = self._send_command("lo")

        if self.laser.is_open:
            print("Closing laser", laser_idx)
            self.laser.close()   
        return
    
    def __enter__(self):
        return self.open()
    
    def __exit__(self, exception_type, exception_value, traceback):
        return self.close()
    
    def get_serial(self):
        command = "gsn?" #look commands in manual
        termination = "\r\n" #always end \r \n
        self.laser.write((command + termination).encode('ascii') ) # encode('ascii')
        result = self.laser.readline().decode('ascii')
        print(result)
        return result

    def quit_and_close(self):
        return self.close()
    
    def get_power(self):
        return self.power