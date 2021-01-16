# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 20
BS = 32

DIRECTORY = 'dataset'
OUTPUT_MODEL = 'gesture_detector.model'
PLOT_FILE = 'plot.png'
CATEGORIES = 4      #how many folders are in the dataset directory

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

# initialize the data and labels
print("[INFO] Loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(DIRECTORY)))
# random.seed(42)
# random.shuffle(imagePaths)

# print(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    # print(label)
    if(label == 'ok'): label = 0
    elif(label == 'ok_left'): label = 1
    elif(label == 'palm'): label = 2
    elif(label == 'palm_left'): label = 3
    labels.append(label)

# print(labels)
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.25, random_state=42)

# convert the labels from integers to verctors
trainY = to_categorical(trainY, num_classes=CATEGORIES)
testY = to_categorical(testY, num_classes=CATEGORIES)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, depth=3, classes=CATEGORIES)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(OUTPUT_MODEL, save_format="h5")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Hand gestures")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(PLOT_FILE)



































# # grab the list of images in our dataset directory, then initialize
# # the list of data (i.e., images) and class images
# print("[INFO] loading images...")


# data = []
# labels = []

# for category in CATEGORIES:
#     path = os.path.join(DIRECTORY, category)
#     for img in os.listdir(path):
#         img_path = os.path.join(path, img)
#         # load the input image (224x224) and preprocess it
#         image = load_img(img_path, target_size=(224, 224))
#         image = img_to_array(image)
#         image = preprocess_input(image)

#         data.append(image)
#         labels.append(category)

# # perform one-hot encoding on the labels
# lb = MultiLabelBinarizer()
# labels = lb.fit_transform(labels)
# # labels = to_categorical(labels)


# data = np.array(data, dtype="float32") / 255.0
# labels = np.array(labels)

# print(data.shape)

# (trainX, testX, trainY, testY) = train_test_split(data, labels,
# 	test_size=0.20, stratify=labels, random_state=42)

# trainY = to_categorical(trainY, num_classes=4)
# testY = to_categorical(trainY, num_classes=4)

# # construct the training image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
# 	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
# 	horizontal_flip=True, fill_mode="nearest")



# # load the MobileNetV2 network, ensuring the head FC layer sets are
# # left off
# baseModel = MobileNetV2(weights="imagenet", include_top=False, 
#     input_tensor=Input(shape=(224, 224, 3)))

# # construct the head of the model that will be placed on top of the
# # base model
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(128, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)

# # place the head FC model on top of the model (this will become
# # the actual model we will train)
# model = Model(inputs=baseModel.input, outputs=headModel)

# # loop over all layers in the base model and freeze them so they will
# # *not* be updated during the first training process
# for layer in baseModel.layers:
#     layer.trainable = False

# # compile our model
# print("[INFO] compiling model...")
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(loss="categorical_crossentropy", optimizer=opt, 
#     metrics=["accuracy"])

# # train the head of the network
# print("[INFO] training head...")
# H = model.fit(
# 	aug.flow(trainX, trainY, batch_size=BS),
# 	steps_per_epoch=len(trainX) // BS,
# 	validation_data=(testX, testY),
# 	validation_steps=len(testX) // BS,
# 	epochs=EPOCHS)

# # make predicitions on the testing set
# print("[INFO] evaluating network...")
# predIdxs = model.predict(testX, batch_size=BS)

# # for each image in the testing set we need to find the index of the
# # label with corresponding largest predicted probability
# predIdxs = np.argmax(predIdxs, axis=1)

# # show a nicely formatted classification report
# print(classification_report(testY.argmax(axis=1), predIdxs,
# 	target_names=lb.classes_))

# # serialize the model to disk
# print("[INFO] saving gesture detector model...")
# model.save("gesture_detector.model", save_format="h5")

# # plot the training loss and accuracy
# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig("plot.png")
