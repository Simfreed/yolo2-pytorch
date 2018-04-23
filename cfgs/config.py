import os
# from .config_voc import *  # noqa
from .config_track import *  # noqa
#from .exps.darknet19_exp1 import *  # noqa
from .exps.track import *

def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


# input and output size
############################
multi_scale_inp_size = [np.array([320, 320], dtype=np.int),
                        np.array([352, 352], dtype=np.int),
                        np.array([384, 384], dtype=np.int),
                        np.array([416, 416], dtype=np.int),
                        np.array([448, 448], dtype=np.int),
                        np.array([480, 480], dtype=np.int),
                        np.array([512, 512], dtype=np.int),
                        np.array([544, 544], dtype=np.int),
                        np.array([576, 576], dtype=np.int),
                        # np.array([608, 608], dtype=np.int),
                        ]   # w, h
multi_scale_out_size = [multi_scale_inp_size[0] / 32,
                        multi_scale_inp_size[1] / 32,
                        multi_scale_inp_size[2] / 32,
                        multi_scale_inp_size[3] / 32,
                        multi_scale_inp_size[4] / 32,
                        multi_scale_inp_size[5] / 32,
                        multi_scale_inp_size[6] / 32,
                        multi_scale_inp_size[7] / 32,
                        multi_scale_inp_size[8] / 32,
                        # multi_scale_inp_size[9] / 32,
                        ]   # w, h
inp_size = np.array([416, 416], dtype=np.int)   # w, h
out_size = inp_size / 32


# for display
############################
def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


base = int(np.ceil(pow(num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(num_classes)]


# detection config
############################
thresh = 0.3


# dir config
############################
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#outdir = '/project/weare-dinner/simonfreedman/cytomod/out'
#DATA_DIR = os.path.join(outdir, 'structs/dens_dens/mkoff0.1_mkend0.1/sd1e7/mdens0.00_xldens1.5/restart_more_motors_long_kend10/seed1e6')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
TRAIN_DIR = os.path.join(MODEL_DIR, 'training')
TEST_DIR = os.path.join(MODEL_DIR, 'testing')

pretrained_model = os.path.join(MODEL_DIR, pretrained_fname)
train_output_dir = os.path.join(TRAIN_DIR, exp_name)
test_output_dir = os.path.join(TEST_DIR, imdb_test, h5_fname)
mkdir(train_output_dir, max_depth=3)
mkdir(test_output_dir, max_depth=4)
trained_model = os.path.join(train_output_dir, h5_fname)

rand_seed = 1024
use_tensorboard = True

log_interval = 50
disp_interval = 10
