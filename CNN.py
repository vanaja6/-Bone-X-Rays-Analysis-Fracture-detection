

from tensorflow.keras import layers from tensorflow.keras import models class CNN(models.Model):
def   init  (self):
super(CNN, self).  init  ()
self.model = models.Sequential([
layers.Conv2D(124, (3, 3), activation='elu',
input_shape=(124, 124, 3)),
layers.MaxPooling2D((2, 2)), layers.BatchNormalization(),
layers.Conv2D(75, (3, 3), activation='elu'),
layers.MaxPooling2D((2, 2)), layers.BatchNormalization(),
layers.Conv2D(64, (3, 3), activation='elu'), layers.Flatten(),
layers.Dense(64, activation='elu'), layers.Dense(2, activation='softmax')
])
def  call (self, inputs): return self.model(inputs)
