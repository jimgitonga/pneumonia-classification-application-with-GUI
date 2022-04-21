
from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

import os
import sys
from os import truncate
from tkinter import *

from tkinter import Frame, Tk, Text, Menu, Label, Entry, Button
from tkinter import filedialog
from tkinter import messagebox as mb
import tkinter as tk
# from tkinter.constants import END

from tkinter import ttk
from PIL import Image
import numpy as np
import cv2


from PIL import ImageFile
import PIL.Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import glob
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from collections import OrderedDict
from torch.autograd import Variable
import PIL
from torch.optim import lr_scheduler
import copy
import json




def get_device():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print(f"Device: {device}")
    return device

get_device()
print("the function finished")
print("the function finished")