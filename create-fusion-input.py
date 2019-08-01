import glob
import scipy.io as sio
import numpy as np
from skimage import img_as_ubyte
from imageio import imwrite

# Parallel libararies
from joblib import Parallel, delayed
import multiprocessing


FILE_PATH = "/home/gwinkelmaier/parvin-labs/nuc-seg/data/Mina/inputs/"

# Get a list of potential field and region images
region_list = glob.glob(FILE_PATH + '*output-region*.mat')
potential_list = glob.glob(FILE_PATH + '*output-potential*.mat')

region_list = sorted(region_list)
potential_list = sorted(potential_list)

# create and save  images
def fcn( patch ):
    # Read in mat files
    R = sio.loadmat(patch[0]);
    P = sio.loadmat(patch[1]);

    # get argmax of region
    # reg = convert_region(R['x'])
    reg = R['x'][1,:,:]
    reg = reg - np.min(reg)
    reg = reg / np.max(reg)
    reg = reg * 255
    # print("min/max: {}/{}".format(np.min(reg), np.max(reg)))

    # Scale potential field
    pot = P['x']
    pot = pot - np.min(pot)
    pot = pot / np.max(pot)
    pot = pot * 255
    pot = np.squeeze(pot)
    # print("min/max: {}/{}".format(np.min(pot), np.max(pot)))

    # Concatenate
    I = np.stack([pot, reg, pot], axis=2).astype(np.uint8)
    # I = img_as_ubyte( I )

    # Create the save name
    save_name = FILE_PATH + patch[0].split('/')[-1] 
    save_name = str.replace(save_name, 'output-region', 'input-fusion')
    save_name = str.replace(save_name, '.mat', '.png')

    imwrite(save_name, I)


# Parallel process the function
num_cores = multiprocessing.cpu_count() / 2
# print("Using {} Cores...".format(num_cores))

Parallel(n_jobs=int(num_cores))(delayed(fcn)(i) for i in zip(region_list, potential_list))
# for i in zip(region_list, potential_list):
#     print("Hello")
#     fcn(i)
#     break
