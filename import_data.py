import os
from natsort import natsorted  # Sort 1, 10, 2, 3 to 1, 2, 3, 10

def data_dirs(expnum, basedir):
    """
    get `.csv` data directory list for specified experiments
    For example, when expNum=1, it returns ["MyData/trial-1", "MyData/trial-2", ..., "MyData/trial-n"]
    """

    experimentfiles = natsorted([file for file in os.listdir(basedir) if 'Experiment' in file])  # only get `Experiment` folders
    experimentfile = experimentfiles[expnum - 1]  # get the file name that you are targeting
    experimentdir = os.path.join(basedir, experimentfile)  # get the data file dir with `join`
    datanames = os.listdir(experimentdir)  # get the data file list

    # get the data file directory lists for a specified `expNum`
    datadirs = []  # make an empty list to contain directories
    for dataname in datanames:
        datadir = os.path.join(experimentdir, dataname)  # dir for a data file
        datadirs.append(datadir)  # append

    # sort in a numerical order
    datadirs = natsorted(datadirs)

    return datadirs
