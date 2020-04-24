from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import functools
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


#        batch_size=12,  # 为了示例更容易展示，手动设置较小的值
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5735,
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True)

    return dataset


AUTOTUNE = tf.data.experimental.AUTOTUNE



data_root = pathlib.Path('D:\\AI\\AI研习社\\102种鲜花分类\\54_data\\train')
all_image_paths = list(data_root.glob('*.jpg'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

# image_count = len(all_image_paths)
# print(image_count)
"""
img_path='D:\\AI\\AI研习社\\102种鲜花分类\\54_data\\train\\1.jpg'
img_raw = tf.io.read_file(img_path)
# print(repr(img_raw)[:100]+"...")

img_tensor = tf.image.decode_image(img_raw)

print(img_tensor.shape)
print(img_tensor.dtype)
"""




path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# print(path_ds)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

"""
plt.figure(figsize=(8,8))
for n, image in enumerate(image_ds.take(4)):
  plt.subplot(2,2,n+1)
  plt.imshow(image)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel('test')
  plt.show()
"""

LABEL_COLUMN = 'label'
LABELS = [0, 1]
raw_train_data = get_dataset('D:\\AI\\AI研习社\\102种鲜花分类\\54_data\\train.csv')
examples, labels = next(iter(raw_train_data))  # 第一个批次
#print("EXAMPLES: \n", examples, "\n")
#print("LABELS: \n", labels)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
#print("LABELS: \n", label_ds)

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
#print(image_label_ds)



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(192, 192)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#images = load_and_preprocess_image(all_image_paths)
#model.fit(images, labels, epochs=10)
model.fit(image_label_ds, epochs=10)
