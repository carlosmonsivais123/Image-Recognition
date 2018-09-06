---
title: "Image Recognition"
author: "Carlos Monsivais"
date: "8/26/2018"
output: html_document
---
#Installing Keras
```{r}
#This is a quick description of what you need to install in order to get the keras and tensorflow packages.

#You need python 3.6 64 bit for this 
#Go to this link https://www.anaconda.com/download/#macos
#install the 64 bit either for windows, linux or mac 
#After this install the following packages
install.packages("devtools")
library(devtools)
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/reticulate")
library(tensorflow)
install_tensorflow()
```

#Loading the packages
```{r}
library(keras)
#You will need to load this library for convolutional layers in the code below along wiht using the one hot encoding function below.

library(EBImage)
#You will need this library in order to read in the images, resize the images and analyze the images.
```

#Reading in the Images
```{r}
setwd("/Users/carlos.monsivais/Desktop")
#You ned to set the working directory as the place where you have all your images, in this case it's my desktop.

pic1 = c("islandlight1.jpeg", "islandlight2.jpeg", "islandlight3.jpeg", "islandlight4.jpeg", "islandlight5.jpeg", "islandlight6.jpeg", "islandlight7.jpeg", "islandlight8.jpeg","tracklight1.jpeg", "tracklight2.jpeg", "tracklight3.jpeg", "tracklight4.jpeg", "tracklight5.jpeg", "tracklight6.jpeg", "tracklight7.jpeg", "tracklight8.jpeg", "minipendant1.jpeg", "minipendant2.jpeg", "minipendant3.jpeg", "minipendant4.jpeg", "minipendant5.jpeg", "minipendant6.jpeg", "minipendant7.jpeg", "minipendant8.jpeg")
#Here we are listing the exact names of the images that we will be using for the training set. You don't put all the images here, only the images that will be used to train the neural network. Make sure to specify the exact name of each image.

train = list()
#Here we are creating an empty list for the training images above. We are doing so because we need to read in the images into this list.

for(i in 1:24) {train[[i]] = readImage(pic1[i])}
#Here we are using a for function where we want it to process the 24 images we have therefore that's why we iterate it 1 to 24 where we use the readImage function to read in the images into the empty list we created called train since thee images are speicifacally the trainin images that we are uploading into R. 

pic2 = c("islandlight9.jpeg", "islandlight10.jpeg","tracklight9.jpeg", "tracklight10.jpeg","minipendant9.jpeg", "minipendant10.jpeg")
#Here we are listing the exact names of the images that we wil be using for our testing set. These images wil only be used to test the model tat we create and see how well it perfomrs, we don't want to use them to train the model, these images should remain untouched.

test = list()
#Here is the empty list we created where we will upload the testing images above. It's the same process as the training images however this time just with the testing images.

for(i in 1:6) {test[[i]] = readImage(pic2[i])}
#Again we are using the for function in order to read in the 6 images therefore we iterate this function from 1 to 6 and use the readImage function to read the images into the empty list called test where we will store the testing images.
```

#Analyzig the Images
```{r}
print(train[[2]])
#This prints out the image matrix along with the dimensions, and the method of storage.

summary(train[[2]])
#Gives us a summary such as the minimum,  1rst quartile, median, mean, 3rd quartile and the maximum of the image matrix.

display(train[[2]])
#Displays the image in the viewer format with your computer as your local host.

plot(train[[2]])
#Displays the image on the plot viewer of R Studio however it uses R Studio as the local host.

par(mfrow = c(3,8))
#This will ensure your plots are in the format of plotting graphs or images in a way so that they are organized in 3 rows by 8 columns format.

for(i in 1:24)plot(train[[i]])
#We are plotting all 24 of the training images hence the iteration of 1 to 24 however we are doing so in a 3 by 8 format because of the code from above.

par(mfrow = c(1,1))
#This makes sure that now when you plot you are doing so in a 1 by 1 format in how we changed it to a 3 by 8 but we want it back to normal and therefore write this part.
```

#Scaling the image
```{r}
str(train)
#Prints out the structure of each of the individual images in the training set. The structure includes the dimensions of the image and part of the image matrix. 

for(i in 1:24){train[[i]] = resize(train[[i]], 100, 100)}
#We are using a for function for the training images hence the 1 to 24 iteration where we are using the resize function on the 24 training images to resize them to 100 by 100 by 3 sizes so that for the convolution layers we can look at equal dimensions.

str(train)
#Looking at the structure of the images to see that they were all resized to the dimensions of 100 by 100.

for(i in 1:6){test[[i]] = resize(test[[i]], 100, 100)}
#We are using the for function for the testing images hence the 1 to 6 iteration where we are using the resize function on  the 6 testing images to resize them to a 100 by 100 by 3 dimension.

train = combine(train)
#This function combines all 24 training images into one image. For example, now the dimensions are 100 by 100 by 3 by 24 meaning that the sizes are still 100 by 100 by 3 and the 24 means that 24 images are combined. We want to combine them so that we can just feed the neural network one whole thing instead of 24 individual things.

x = tile(train, 8)
#Here we are putting the 24 training images into a format where it will generate a single image. The number 8 tells the function how many frames per row. In this case we want 8  frames or 8 images per row creating the one image.

display(x, title = "Training Pictures")
#Now we are displaying the combined image made up of our 24 training images. We have given it the title Training Pictures.

test = combine(test)
#This function combines all 6 testing images into one image. For example, now the dimensions are 100 by 100 by 3 by 6 meaning that the sizes are still 100 by 100 by 3 and the 24 means that 6 images are combined.

y = tile(test, 2)
#Here we are putting the 6 testing images into a format where it will generate a single image.

display(y, title = "Testing Pictures")
#Now we are displaying the combined image made up of our 6 testing images. We have given it the title Testing Pictures.

str(train)
#Here is the structure of the 24 training images. We can see that now that we transformed the 24 images into 1 we only get one output from this function with the dimensions of 100 by 100 by 3 by 23.

str(test)
#Here is the structure of the 6 testing images. We can see that now that we transformed the 6 images into 1 we only get one output from this function with the dimensions of 100 by 100 by 3 by 23.
```

#Reordering the Dimensions
```{r}
dim(train)
#These are the dimensions of the training image.

train = aperm(train, c(4, 1, 2, 3))
#We are using the aperm function because we want to change the dimension of the training image to go from the original dimensions 100 by 100 by 3 by 24 to these new dimensions being 24 by 100 by 100 by 3.

str(train)
#Here is the new structure of the training image which should be 24 by 100 by 100 by 3.

dim(test)
#These are the dimensions of the testing image.

test = aperm(test, c(4,1,2,3))
#We are using the aperm function because we want to change the dimension of the testing image to go from the original dimensions 100 by 100 by 3 by 6 to these new dimensions being 6 by 100 by 100 by 3.

str(test)
#Here is the new structure of the training image which should be 6 by 100 by 100 by 3.
```

#Response
```{r}
trainy = c(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2)
#Here we are assigning numerical values to the training images. For example, the 0's correspond to the islandlight images, the 1's correspond to the tracklight images and the 2's correspond to the minipendant images in the training data set.

testy = c(0,0,1,1,2,2)
#Here we are assigning numerical values to the testing images. For example, the 0's correspond to the islandlight images, the 1's correspond to the tracklight images and the 2's correspond to the minipendant images in the testing data set.
```

#One Hot Encoding
```{r}
trainLabels = to_categorical(trainy)
#Here we are using the one hot encoding method to reorganize how the computer sees the format of the training images. The one hot encoding format makes it easier to work with for neural networks.

testLabels = to_categorical(testy)
#Here we are using the one hot encoding method to reorganize how the computer sees the format of the testing images. The one hot encoding format makes it easier to work with for neural networks.
```

#Creating the Model
#Fix here
```{r}
model = keras_model_sequential()
#Here we are creating an empty function so that we can insert the model that we are creting below. We use the keras_model_sequential becuase we want to create a model using a layered convolutional network.

model %>%
#We are creating the convolutional network under the name model. The %>% means we will continue creating the model on the next line however it still belongs to the variable model.
  
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(100,100,3)) %>%
  #We are using the layer_conv_2d function to create the convolutional layer. The input shape is the dimensions of the training images. We reshaped them to be 100 by 100 by 3 as        specified above. 
  
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu') %>%
  #Again we create another convolutuional layer however we leave off the input_shape since we already did that in the convolutional layer above.
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  #To reduce the size of the information we use this pooling layer. 
  
  layer_dropout(rate = 0.25) %>%
  #We use this layer dropout function to remove 25% of the images in the training data set so that we don't overfit.
  
  layer_conv_2d(filters  =64, kernel_size = c(3,3), activation = 'relu') %>%
  #We are creating another convolutional layer however this time with 64 filters meaning that it will recognize 64 features based on the image. The first features are not as detailed   however the later features recognize more and more detailed features belonging to the image. 
  
  layer_conv_2d(filters  =64, kernel_size = c(3,3), activation = 'relu') %>%
  #Here is another convolutional layer again, with 64 filters.
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  #We are using a pooling layer to reduce the size of the information.
  
  layer_dropout(rate = 0.25) %>%
  #Again, we are removing 25% of the training image in order to not overfit in the model.
  
  layer_flatten() %>%
  #By using the flattening layer we are converting this 3 dimensional image into a 1 dimensional object.
  
  layer_dense(units = 256, activation = 'relu') %>%
  #How many neuron we want our 1 dimensional transformation to have.
  
  layer_dropout(rate = 0.25) %>%
  #Again removing 25% of the training data so that we don't overfit.
  
  layer_dense(units = 3, activation = 'softmax')%>%
  #Here we have 3 units because we have 3 categories. Therefore, we want to have the same amount of unit as categories. We use the softmax function because we want to be able to see    the results in the forms of probabilities so that we can understand them.
  
  compile(loss = 'categorical_crossentropy', optimizer = optimizer_sgd(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov =TRUE), metrics = c('accuracy'))
  #Compiling the model to update its parameters and to prepare it for the next phase of training.

summary(model)
#Gives a summary of the model that includes the total number of parameters at each phase of convolutions, pooling, dropouts, flattening, and using the dense function (this gave us a 1 dimensional view).
```

#Fitting the Model
```{r}
history = model %>% 
  #Here we will fit the model to the training data however we will call the fitted model history.
  
  fit(train, trainLabels, epochs = 60, batch_size = 32, validation_data = list(test, testLabels))
  #We are using the fit function to fit the model to the training images that we have. We are using the training image matrixes and the one hot encoding matrix to train the model. The   validation step is to see the loss function when we plot this model however it does not use the testing images to train the model, just to see the loss function it will create. Here   the number of epochs is the number of iterations the model runs through and this outputs an interactive graphing viewer letting you slide your mouse through the graphs.

plot(history)
#Plots the models accuracy and loss function in the Plot viewer in R Studio.
```

#Evaluation and Prediction Training
```{r}
model %>% evaluate(train, trainLabels)
#Now we are using the model to evaluate the training data. Even though we used the training data to fit and create the model just out of curiosity I want to see how well the model  did using the images it was trained with. By using the evaluate function we are seeing how well we can predict the training image's matrix and one hot encoding matrix.

predtrain = model %>% predict_classes(train)
#We are using the predict_classes to predict the classes of the training images, in this case whether it's an islandlight, tracklight, or a minipendant.

table(Predicted = predtrain, Actual = trainy)
#We are putting the predictions and the actual results on a table to compare how well our model did.

prob = model %>%
#Here we are creating a variable called prob where we will see the probabilities assigned to the predictions of each class. 
  
  predict_proba(train)
  #The predict_proba function is used to show the predicted probabilities of assigning each image to its category.

cbind(round(prob,4), Predicted_class = predtrain, Actual = trainy)
#I am now combining the columns of the predicted probabilities, the predicted class, and the actual class to get a good understanding of what the model outputted. I am also rounding the probabilities to the 4th decimal to not have massive values.
```

#Evaluating and Predicting Testing
```{r}
model %>% evaluate(test, testLabels)
#Now we are using the model to evaluate the testing data. By using the evaluate function we are seeing how well we can predict the testing image's matrix and one hot encoding matrix.

predtest = model %>% predict_classes(test)
#We are using the predict_classes to predict the classes of the testing images, in this case whether it's an islandlight, tracklight, or a minipendant.

table(Predicted = predtest, Actual = testy)
#We are putting the predictions and the actual results on a table to compare how well our model did.

prob = model %>%
#Here we are creating a variable called prob where we will see the probabilities assigned to the predictions of each class.   
  
  predict_proba(test)
  #The predict_proba function is used to show the predicted probabilities of assigning each image to its category.

cbind(round(prob,4), Predicted_class = predtest, Actual = testy)
#I am now combining the columns of the predicted probabilities, the predicted class, and the actual class to get a good understanding of what the model outputted. I am also rounding the probabilities to the 4th decimal to not have massive values.
```

