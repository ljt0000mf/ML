import os
import numpy as np
import pandas as pd
import matplotlib.pylab as pylab
import matplotlib.image as mplim
from sklearn.metrics import accuracy_score
import  torch
from torch.autograd import Variable

seed = 128
rng = np.random.RandomState(seed)

root_dir = os.path.abspath('D:\\AI\\AI研习社\\102种鲜花分类\\54_data')
data_dir = os.path.join(root_dir, 'train')
# print(data_dir)

train = pd.read_csv(os.path.join(root_dir, 'train.csv'))
# print(train.head())

img_name = rng.choice(train.filename)
filepath = os.path.join(data_dir, img_name)

# scipy.ndimage.imread
# img = pilim.open(filepath)
img = mplim.imread(filepath)

"""
pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()
"""
temp = []
for img_name in train.filename:
    image_path = os.path.join(root_dir,'train',img_name)
    img = mplim.imread(image_path)
    img = img.astye('float32')
    temp.append(img)


train_x = np.stack(temp)
train_x /= 255.0

input_num_units = 28*28
hidden_num_units = 50
output_num_units = 102

epochs = 5
batch_size = 128
learning_rate = 0.001

model = torch.nn.Sequential(
    torch.nn.Linear(input_num_units, hidden_num_units),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_num_units, output_num_units)
)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def preproc(unclean_batch_x):
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    return temp_batch


def batch_creator(batch_size):
    dataset_name = 'train'
    dataset_length = train_x.shape[0]
    batch_mask = rng.choice(dataset_length, batch_size)
    batch_x = eval(dataset_name + '_x')[batch_mask]
    batch_x = preproc(batch_x)

    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values

     return batch_x,batch_y


total_batch = int(train.shape[0]/batch_size)

for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_x, batch_y = batch_creator(batch_size)

        x, y = Variable(torch.from_numpy(batch_x)), Variable(torch.from_numpy(batch_y),requires_grad=False)
        pred = model(x)

        loss = loss_fn(pred,y)

        loss.backward()
        optimizer.step()
        avg_cost += loss.data[0]/total_batch

    print(epoch,avg_cost)

x  = Variable(torch.from_numpy(preproc(train_x)),requires_grad=False)
pred = model(x)

final_pred = np.argmax(pred.data.numpy(), axis=1)

print("final_pred:",final_pred)








