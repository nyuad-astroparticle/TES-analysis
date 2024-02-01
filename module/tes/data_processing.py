# import libraries
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
import pyperclip
import datetime
import seaborn as sns

from matplotlib.patches import ConnectionPatch
from matplotlib.colors import ListedColormap
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from scipy import signal
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans
from sklearn import manifold
from matplotlib.dates import DateFormatter

from readTrc import Trc

class FileNotFoundError(Exception):
        pass

class Data_Processing:
    def __init__(self, traces_path, channel = 'C1'):
        
        # Oscilloscope channel on which the data was recorded
        self.ch = channel
        self.trigger = 0.0
        self.set_trigger(traces_path, self.ch)
        
        # Oscilloscope output that consists of: time, signal, metadata
        self.traces = None
        self.load_traces(traces_path)

        # What a photon pulse should look like
        self.reference_pulses = None
        self.load_reference_pulses()

        # Voltage from reference pulse
        self.REFERENCE_VOLTAGE = None
        self.set_reference_voltage()

        #Filtering for trigger in traces
        # Indexes of triggered traces  
        self.the_data_index = None
        # Triggered traces
        self.the_data = None
        self.set_the_data()


    def load_traces(self, data_dir):
        """
        Loads traces from the specified directory for each channel.

        Args:
        data_dir (str): The directory containing the data files.

        Returns:
        None: The method updates the `traces` attribute of the class.
        """
        channels = ['C1', 'C2', 'C3', 'C4']

        # Load the data per channel
        filenames = {channel: [file for file in os.listdir(data_dir) if channel in file] for channel in channels}
        # Sort filenames for proper alignment
        for channel in channels:
            filenames[channel] = np.sort(filenames[channel])

        # Load the data
        trc_loader = Trc()  # Create a loader object
        self.traces = {channel: [trc_loader.open(os.path.join(data_dir, filename)) for filename in filenames[channel]] for channel in channels}

        # Example output to verify data loading
        ch = self.ch
        if ch in self.traces:
            print('Traces for channel', self.ch ,'loaded:',len(self.traces[ch]))
            print('Aquisition start time:', self.traces[ch][0][2]['TRIGGER_TIME'])
            print('End time:', self.traces[ch][-1][2]['TRIGGER_TIME'])
            print('Total time of the run (hh:mm:ss):', self.traces[ch][-1][2]['TRIGGER_TIME'] - self.traces[ch][0][2]['TRIGGER_TIME'])
        else:
            print(f"No data found for channel {ch}")

    def get_csv_path(self, filename):
        """
        Provides a path for loading references. Need when used as a module.

        Args:
        filename (str): name of the file (currently used only for reference.dat)

        Returns:
        csv_path (str): full path to the file

        """
        # Get the directory of the current module
        module_dir = os.path.dirname(__file__)
        # Construct the full path to the CSV
        csv_path = os.path.join(module_dir, filename)
        return csv_path

    def load_reference_pulses(self):
        """
        Loads reference pulses from a specified file.

        Args:
        file_path (str): The path to the file containing the reference pulses.

        Returns:
        None: The method updates the `reference_pulses` attribute of the class.

        """
        try:
            self.reference_pulses = pd.read_csv(self.get_csv_path("reference.dat"), skiprows=5, header=[0, 1], sep='\t')
            print("Reference pulses loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading reference pulses: {e}")

    def set_reference_voltage(self):
        """
        
        Sets reference_voltage 

        Args:
        None: requires reference_pulses and traces loaded

        Returns:
        None: The method sets the reference_voltage

        """
        self.REFERENCE_VOLTAGE  = signal.resample(self.reference_pulses['Amp_2ph_1.6eV'].to_numpy().T[0],len(self.traces[self.ch][0][0]))

    def extract_number_before_mV(self, s):
        """
        Extracts the number before the occurrence of 'mV' in the given string.

        Args:
        s (str): The string from which the number is to be extracted.

        Returns:
        float: The extracted number. Returns None if 'mV' is not found or if the number is not properly formatted.
        """
        try:
            # Find the index of 'mV' in the string
            mV_index = s.index('mV')
            
            # Start from mV_index and move backwards to find the start of the number
            start_index = mV_index
            while start_index > 0 and (s[start_index - 1].isdigit() or s[start_index - 1] == '-' or s[start_index - 1] == '.'):
                start_index -= 1

            # Extract the number substring
            number_str = s[start_index:mV_index]

            # Convert to float and return
            return -1.0 * abs(float(number_str))
        except (ValueError, IndexError):
            return None

    def find_file_starting_with(self, directory, start_str):
        """
        Finds the first file in the given directory that starts with the specified string.
        Raises an error if no such file is found.

        Args:
        directory (str): The directory to search in.
        start_str (str): The starting string of the file name to find.

        Returns:
        str: The name of the first matching file.

        Raises:
        FileNotFoundError: If no file starting with the specified string is found.
        """
        # Check if the directory exists
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"The directory {directory} does not exist.")

        # Iterate over the files in the directory
        for file in os.listdir(directory):
            if file.startswith(start_str):
                return file

        # Raise an error if no file is found
        raise FileNotFoundError(f"Channel {start_str} was not found in the given directory.")
    
    def set_trigger(self, directory, channel):
        """
        Set the trigger for selected channel from filename
        
        Args:
        directory (str): The directory of Trc files
        channel (str): which channel trigger to find

        Returns:
        None: Sets the trigger
        """
        
        self.trigger = (self.extract_number_before_mV(self.find_file_starting_with(directory, channel)) + 1.0)/1000
        print('Trigger for channel', self.ch, 'is set to:', self.trigger*1000, 'mV')

    def set_the_data(self):
        """
        Select the traces that had nonzero signal (can add ch3 and ch4 in case we need PMT coincidence as well)
        
        Args:
        None: uses trigger and traces

        Returns:
        None: sets the data (2d-array [times][voltages])

        """
        mask_ch0            = [np.min(data[1]) < self.trigger for data in self.traces[self.ch]]
        self.the_data_index = np.where(np.array(mask_ch0))[0]
        self.the_data       = [self.traces[self.ch][i] for i in self.the_data_index]

        passed = np.unique(mask_ch0,return_counts=True) # how many signal passed the trigger (True)
        not_triggered = 0
        triggered = 0
 
        if len(passed[0]) == 1:
            if passed[0][0]: triggered = passed[1][0]
            else: not_triggered = passed[1][0]

        else:
            not_triggered = np.unique(mask_ch0,return_counts=True)[1][0]
            triggered = np.unique(mask_ch0,return_counts=True)[1][1]
            
        print('Number of signals triggered:', triggered, '\nNot triggered:', not_triggered)

    def plot_all_triggered(self):
        """
        Plots all of the triggered pulses

        Args:
        None

        Returns:
        None: simply plots
        
        """
        plt.figure()
        for i in self.the_data_index:
            plt.plot(self.traces[self.ch][i][1])
        plt.ylabel("Voltage")
        plt.title(f"{self.ch} triggered events")
        plt.show()