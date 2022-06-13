import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

base_path="C:/facedetect/"

classifier = Sequential()
classifier.add(Convolution2D(64,(2,2),input_shape=(64,64,1),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(128,activation = 'relu'))      # first and input layer
classifier.add(Dense(256,activation = 'relu'))      # first and hidden layer
classifier.add(Dense(512,activation = 'relu'))      # second and hidden layer
classifier.add(Dense(4,activation = 'softmax'))     #output=4

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])   

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2, #shearing transformation
        zoom_range=0.1,       # zoom in required
        horizontal_flip=True) # if the images data needs to be horizontally flipped, applicable for real world images

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(base_path+"final",
        target_size=(64,64), #size of the image in the model 
        batch_size=8,
        color_mode='grayscale',
        class_mode='categorical')

#creates the test set
test_set = test_datagen.flow_from_directory(
        base_path+'test',
        target_size=(64,64),       #size of the image in the model
        batch_size=8,
        color_mode='grayscale',
        class_mode='categorical')
classifier.fit_generator(train_set,
                    steps_per_epoch=50,     #number of images
                    epochs=10,
                    validation_data=test_set,
                     validation_steps=5)

train_set.class_indices
classifier.save("model.h5")