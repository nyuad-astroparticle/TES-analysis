# import libraries
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
import pyperclip
import datetime
import cupy as cp
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


from data_processing import Data_Processing

class PCA(Data_Processing):

    def __init__(self, traces_path, channel = 'C1'):
        super().__init__(traces_path, channel)
        
        self.N_CLUSTERS = None
        self.X_compressed = None


    def plot_signal(voltage,fig=None,ax=None,order=2,mysize=1/50,**kwargs):
        '''
        Returns a Pyplot figure of the smoothed voltage signal as a function of
        event number

        Args:
        voltage (arr): voltage of a signal from the data
        ----------------------------------------------------------------------------Artem needs to elaborate on the arguments here

        Returns:
        fig,ax

        '''

        sos=signal.butter(order,mysize, output='sos')
        butterybiscuits=signal.sosfilt(sos,voltage)
        
        if fig is None: fig = plt.figure()
        if ax is None: ax  = fig.add_subplot(111)
        
        ax.plot(butterybiscuits,**kwargs)
        
        return fig,ax
    
    def values_of_interest(self,time,data,order,mysize):
        """
        These are functions used as dimentions in the PCA

        ----------------------------------------------------------------------------Artem please write the description of the function, args, and return

        """
        sos=signal.butter(order,mysize, output='sos')
        butterybiscuits=signal.sosfilt(sos,data)
        
        peak            = np.argmin(butterybiscuits)
        area            = np.sum(butterybiscuits[:-1]*(time[1:]-time[:-1]))
        peakheight      = np.min(butterybiscuits)
        halfheights     = np.where(butterybiscuits-peakheight/2<0)[0]
        fwhm            = 0 if len(halfheights)<1 else time[halfheights[-1]]-time[halfheights[0]]
        noize           = len([x for x in data if x>0.00005])
        medheight       = np.median(butterybiscuits)
        meanheight      = np.mean(butterybiscuits)
        meantime        = np.mean(np.abs(butterybiscuits)*(time))/np.sum(np.abs(butterybiscuits[:-1])*(time[1:]-time[:-1]))
        std             = np.std(butterybiscuits)
        grad            = (butterybiscuits[1:]-butterybiscuits[:-1])/(time[1:]-time[:-1])
        grad_min        = np.mean(grad[np.where(grad<0)])
        grad_max        = np.mean(grad[np.where(grad>0)])
        corr            = signal.correlate(butterybiscuits/np.min(butterybiscuits)*np.min(self.REFERENCE_VOLTAGE),self.REFERENCE_VOLTAGE)
        area_cumulative = np.cumsum(butterybiscuits[:-1]*(time[1:]-time[:-1]))
        cumt_percentage = area_cumulative/area
        percentage      = 0.97
        time_at_98_perc = time[np.argmax(cumt_percentage >= percentage)]
        time_under_trig = len([x for x in butterybiscuits if x >= -0.033])
        
        return (time_under_trig, noize, area,area/std, peakheight, time_at_98_perc, meantime, np.sqrt(np.mean(grad**2)), grad_min, grad_max, np.max(corr))

    def feature_reduction(self):
        """
        This function performs Principal component analysis on the selected functions together with feature reduction and signle value decomposition.

        Args:
        None: works with the data

        Returns:
        None: saves self.X_compressed for future uses

        """
        
        # Get the data for each of the pulses
        values      = cp.array([self.values_of_interest(*data[:2],order=2,mysize=1/30) for data in tqdm(self.the_data)])

        # Center data
        mu          = cp.tile(np.mean(values,axis=0),len(values)).reshape(values.shape)
        X_centered  = (values-mu)
        X_centered  = X_centered / cp.tile(np.max(X_centered,axis=0)-cp.min(X_centered,axis=0),len(X_centered)).reshape(X_centered.shape)

        # Single Value Decomposition
        u,s,vT      = cp.linalg.svd(X_centered)

        # plot the singlar values for the  D  matrix.
        # 1. Calculate the D matrix using s: D is s*s
        D       = cp.asnumpy(s*s)
        labels  = range(len(X_centered[0]))

        # 2. Set the fig size to (15,5)
        fig     = plt.figure(figsize=(10,5))
        ax      = fig.add_subplot(111)

        # 3. Add the line chart using plt.plot( ?? ,'bo-')
        ax.plot(labels,D,'bo-')

        # 3. Add proper tital, ticks, axis labels
        ax.set_title('Singular Values of Datapoints')
        ax.set_xlabel('Label for Linear Combination of (PCs)')
        ax.set_ylabel('Singular Value')
        ax.set_xticks(labels)
        # ax.set_yscale('log')

        # Obtaining our compressed data representation:
        # 1. Determine at least k singular values that are needed to represent the data set from the fig above
        k = np.array(list(labels))

        # 2. Obtain the first k of v^T and store it
        v = vT[k].T

        # 3. Calculate the compressed data using np.matmul(), X and stored first k of v^T
        self.X_compressed = cp.matmul(X_centered,v).get()

        # 4. Print the compressed value of X
        #print(v)

        plt.show()


    def KMeans_clustering(self, N_CLUSTERS):

        """
        ----------------------------------------------------------------------------
        """
        
        # argument N_Clusters 

        # Enable '%matplotlib widget' if running in Jupyter Notebook
        if 'ipykernel' in sys.modules and 'IPython' in sys.modules:
            from IPython import get_ipython
            get_ipython().run_line_magic('matplotlib', 'widget')
        
        # PARAMETERS
        pyper       = []

        sns.set_palette("colorblind")
        # Get the centers
        DATA    = self.X_compressed[:,[0,1]]

        kmeans  = KMeans(n_clusters=N_CLUSTERS, n_init="auto").fit(DATA)
        closestpoints=[]

        fig     = plt.figure(figsize = (14,7))
        ax1     = fig.add_subplot(121)
        ax2     = fig.add_subplot(122)
        # Create a colormap from the Seaborn palette
        colorblind_palette = sns.color_palette("colorblind")
        cmap = ListedColormap(colorblind_palette)
        COLORS  = np.array([kmeans.predict([x[:2]])[0] for x in DATA])
        norm    = matplotlib.colors.Normalize(vmin=COLORS.min(),vmax=COLORS.max())
        points  = ax1.scatter(*(DATA.T),s=10,alpha=0.5,c=cmap(norm(COLORS)))
        centers = ax1.scatter(*kmeans.cluster_centers_.T,color='k',marker='x',s=50)
        ax1.set_title('Clusters')
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Voltage [V]')
        IDX_SELECTED  = []

        cursor_p = mplcursors.cursor(points)

        def callback(sel):
            """
            ----------------------------------------------------------------------------
            """
            I = sel.index
            cursor_p.visible = True
            cursor_c.visible = False
            sel.annotation.set_text(sel.index)
            fig.suptitle(f'Idx {I}')
            ax2.clear()
            pyper.append(I)

            self.plot_signal(self.the_data[I][1],fig=fig,ax=ax2,order=0,mysize=1/50,color='k',alpha=0.2,lw=0.5)
            self.plot_signal(self.the_data[I][1],fig=fig,ax=ax2,order=2,mysize=1/50,color='k',lw=1)
            ax2.plot(self.REFERENCE_VOLTAGE*self.the_data[I][1].min()/self.REFERENCE_VOLTAGE.min(), color='deepskyblue',lw=1,ls=(0,(5,10)))
            sel.annotation.get_bbox_patch().set(fc="white",lw=0,alpha=0)
            sel.annotation.arrow_patch.set(arrowstyle="]-", fc="k")
            CC = ['k']*9
            CC[COLORS[I]] = 'deeppink'
            centers.set_color(CC)

            # val    = values[I]
            # function_name = ["fwhm", "peakheight/area", "time_at_98_perc", "(meantime-time[peak])/peakheight", "std", "area", "np.sqrt(np.mean(grad**2))", "grad_min", "grad_max", "np.max(corr)"]
            # max_name_width = max(len(name) for name in function_name)
            # print("-" * (max_name_width + 30))
            # print(f"{[I]},{'Name':<{max_name_width}}\tResult")
            # for i, name in enumerate(function_name):
            #     result = val[i]
            #     print(f"{name:<{max_name_width}}\t{result}")
            
        cursor_p.connect("add", lambda sel: callback(sel))


        cursor_c = mplcursors.cursor(centers)

        def callback_center(self, sel):
            """
            ----------------------------------------------------------------------------
            """
            I = sel.index
            idx_c = np.where(COLORS==I)[0]
            sel.annotation.set_text(len(idx_c))
            sel.annotation.get_bbox_patch().set(fc="white",lw=0,alpha=0)
            sel.annotation.arrow_patch.set(arrowstyle="wedge", fc="k")
            fig.suptitle(f'Center {I}')
            ax2.clear()
            cursor_p.visible = False
            cursor_c.visible = True
            #pyperclip.copy(", ".join([str(i) for i in idx_c]))
            #PYPER = eval("[" + ", ".join([str(i) for i in idx_c]) + "]")
            pyper.append(idx_c)
            print(len([str(i) for i in idx_c]))
            print((", ".join([str(i) for i in idx_c])))
            CC = ['k']*9
            CC[I] = 'cornflowerblue'
            centers.set_color(CC)

            ax2.set_title(f'Number of Points {len(idx_c)}')
            for i in idx_c:
                #plot_signal(the_data[I][1],fig=fig,ax=ax2,order=0,mysize=1/50,color='k',alpha=0.2,lw=0.5)
                self.plot_signal(self.the_data[i][1],fig=fig,ax=ax2,order=2,mysize=1/50,lw=1,label=f'{i}')
            
        cursor_c.connect('add', lambda sel:callback_center(sel))