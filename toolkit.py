import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from cafca.hdf.api import Database
from cafca.util import audio, playlist, playthrough, show
from cafca.learn import Model, DefaultHP, DEVICE, numcpu, to_torch
from cafca.learn.modules import Pass, ParamedSampler, Abs
from cafca.learn.losses import weighted_L1
from cafca.learn.data import load

from cafca.transform.time import stretch
from cafca.transform.pitch import shift, steps2rate as s2r

from cafca.algos import *


