import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms, utils
from torchvision import models as tm

preresult = []
preresult.append(['img',1])
column = ['filename', 'label']
resultfile = pd.DataFrame(columns=column, data=preresult)
resultfile.to_csv('D:\\AI\\AI研习社\\102种鲜花分类\\54_data\\preresult.csv')
print('save End')

imgnet = tm.densenet201(True)
imgnet
