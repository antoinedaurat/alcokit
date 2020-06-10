import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import h5py

from cafca.util import audio, playlist, playthrough, show
from cafca.extract.segment import SegmentList
from cafca.hdf.api import Database, Fetcher
from cafca.learn import Model, DefaultHP, DEVICE
from cafca.learn.utils import to_torch, numcpu
from cafca.learn.modules import Pass, ParamedSampler, Abs
from cafca.learn.losses import weighted_L1

from cafca.transform.time import stretch
from cafca.transform.pitch import shift, steps2rate as s2r

from cafca.algos import *

