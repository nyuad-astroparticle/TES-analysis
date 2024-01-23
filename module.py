import sys

def loadModules():
    """
    Attempts to import necessary modules for the script.

    This function imports a set of modules required for running the analysis.
    If a module is not installed, it raises an ImportError with a message
    indicating which module needs to be installed. If this function is run
    within a Jupyter Notebook, it also enables the '%matplotlib widget' magic
    command for interactive plotting.
    """
    required_modules = {
        "os": None,
        "numpy": "np",
        "matplotlib": None,
        "matplotlib.pyplot": "plt",
        "pandas": "pd",
        "mplcursors": None,
        "pyperclip": None,
        "datetime": None,
        "cupy": "cp",
        "seaborn": "sns",
        "readTrc": "Trc",
        "scipy.signal": "signal",
        "tqdm.notebook": "tqdm",
        "sklearn.cluster": "KMeans",
        "sklearn": "manifold",
        "matplotlib.patches": ["ConnectionPatch", "Rectangle"],
        "matplotlib.colors": "ListedColormap",
        "matplotlib.widgets": "RectangleSelector",
        "matplotlib.dates": "DateFormatter"
    }

    for module, alias in required_modules.items():
        try:
            if isinstance(alias, list):
                for sub_module in alias:
                    exec(f"from {module} import {sub_module}")
            elif alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
        except ImportError:
            raise ImportError(f"The '{module}' module is not installed. Please install it to proceed.")

    # Enable '%matplotlib widget' if running in Jupyter Notebook
    if 'ipykernel' in sys.modules and 'IPython' in sys.modules:
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'widget')

loadModules()


import os
import numpy as np
from readTrc import Trc

class TesAnalysis:
    def __init__(self):
        self.traces = None
        self.reference_pulses = None

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
        ch = 'C1'
        if ch in self.traces:
            print(len(self.traces[ch]))
            print(self.traces[ch][0][2]['TRIGGER_TIME'])
            print(self.traces[ch][-1][2]['TRIGGER_TIME'])
            print(self.traces[ch][-1][2]['TRIGGER_TIME'] - self.traces[ch][0][2]['TRIGGER_TIME'])
        else:
            print(f"No data found for channel {ch}")

    def load_reference_pulses(self, file_path):
        """

        I NEED THE REFERENCE PULSE DATA FILE!!!!
        Loads reference pulses from a specified file.

        Args:
        file_path (str): The path to the file containing the reference pulses.

        Returns:
        None: The method updates the `reference_pulses` attribute of the class.
        """
        try:
            self.reference_pulses = pd.read_csv(file_path, skiprows=5, header=[0, 1], sep='\t')
            print("Reference pulses loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading reference pulses: {e}")