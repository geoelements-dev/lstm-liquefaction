import os
from natsort import natsorted # Sort 1, 10, 2, 3 to 1, 2, 3, 10

def get_data_dir_list(expnum, basedir):
    """
    get `.txt` data directory list for specified experiments
    For example, when expnum=1, it returns ["MyData/trial-1", "MyData/trial-2", ..., "MyData/trial-n"]
    """
    
    experimentlist = os.listdir(basedir)  # get a list of file names in `basedir`
    experimentlist = [ex for ex in experimentlist if 'Experiment' in ex]  # only get `Experiment` folders
    experimentlist = natsorted(experimentlist)  # `natsorted` enables to sort "1, 10, 2, ..." to "1, 2, ..., 10, ..."
    expfoldername = experimentlist[expnum - 1]  # get the file name (e.g, "Trial-5_accel_corr_Motion12_300mv.txt")
    expdir = os.path.join(basedir, expfoldername)  # get the data file dir with `join`
    dataList = os.listdir(expdir)  # get the data file list
    
    # get the data file directory lists for a specified `expnum`
    datadirlist = []  # make an empty list to contain directories
    for data in dataList:
        datadir = os.path.join(expdir, data)  # dir for a data file
        datadirlist.append(datadir)  # append
    
    # sort in a numerical order
    datadirlist = natsorted(datadirlist)
    
    return datadirlist
