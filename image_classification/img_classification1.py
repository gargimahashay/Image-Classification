import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from skimage.transform import resize
plt.style.use('fivethirtyeight')


from keras.datasets import cifar10
# two tuple for training datasets here it starts downloading
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# data type of the variable
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))


print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# look at image at first
# index = int(input("Enter the index :"))
index = 10

print(x_train[index])
# as a picture format

# show the image as a picture
# img = plt.imshow(x_train[index])
# img = cv2.imshow(x_train[index])



# get the label of image

print('The image label is:', y_train[index])

# get the image classification

classification = ['aeroplane', 'automobile', 'bird', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print('The image class is:', classification[y_train[index][0]])

# convert the labels into a set of 10 numbers to input into the neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

print(y_train_one_hot)

print('the one hot label is:', y_train_one_hot[index])

#normalize the pixel value between 0 and 1

x_train = x_train/255
x_test = x_test/255

print(x_train[index])

# model architecture
model = Sequential()

# add the first layer
model.add(Conv2D(32, (5, 5), activation = 'relu', input_shape=(32, 32, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))


# add a flattenign layer

model.add(Flatten())
# add a layer with 1000 neurons
model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.5))
# add a layer with 500 neurons
model.add(Dense(500, activation='relu'))

# add a drop out layer
model.add(Dropout(0.5))

model.add(Dense(250, activation='relu'))

model.add(Dense(10, activation='softmax'))

# compiling starts here
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# train the model

hist = model.fit(x_train, y_train_one_hot, batch_size=256, epochs = 10, validation_split= 0.2)

# model.save('C://Users//HP//Desktop//Aksh//Extra//New folder//my_practice_one//img_classification.model')

# evaluate model with test
model.evaluate(x_test, y_test_one_hot)[1]

# visualize accuracy of model
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Checking model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()



# visualize loss of model
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Checking model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


#test model with example

# photo = askopenfilename()
img = cv2.imread('cat.jpg')
img = cv2.resize(img, (32, 32, 3))


# ig = cv2.imshow(img)

predictions = model.predict(np.array([resized_image]))

predictions

list_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp

print(list_index)

# print the first 5 predications
for i in range(5):
    print(classification[list_index[i]], ':', round(predictions[0][list_index[i]] * 100, 2), '%')
