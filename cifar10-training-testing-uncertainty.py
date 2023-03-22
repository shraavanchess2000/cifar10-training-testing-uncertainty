#####IMPORTS#####

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from keras.layers.core import Lambda
from keras import backend as K
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report

#####FUNCTIONS#####

def color_map(label):
    if label == 0:
        return 'red'
    if label == 1:
        return 'pink'
    if label == 2:
        return 'orange'
    if label == 3:
        return 'brown'
    if label == 4:
        return 'yellow'
    if label == 5:
        return 'green'
    if label == 6:
        return 'blue'
    if label == 7:
        return 'violet'
    if label == 8:
        return 'gray'
    if label == 9:
        return 'black'

def create_dropout_model():
        #In the style of VGG, which is a proven and high-performing network for image recognition and classification, this model consists of multiple blocks that each contain two convolutional layers and one max-pooling layer, with convolutional layers in each successive block containing double the number of filters as in the previous one. Meanwhile, dropout layers will not only prevent overfitting, but will help quantify uncertainty later in this project. Only 20% of units will be dropped by each dropout layer, but because there are are multiple such dropout layers, overfitting is still mitigated.
    
    model = Sequential()

    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),padding='same'))
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same'))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
    model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))
    model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
    model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))
    model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
    model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dense(10,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.001, momentum=0.9),metrics=['accuracy'])
    
    return model

def create_permadropout_model():
    #PermaDropout code thanks to https://stackoverflow.com/questions/47787011/how-to-disable-dropout-while-prediction-in-keras
    
    def PermaDropout(rate):
        return Lambda(lambda x: K.dropout(x, level=rate))

    model = Sequential()

    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),padding='same'))
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(PermaDropout(0.2))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same'))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(PermaDropout(0.2))
    model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
    model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(PermaDropout(0.2))
    model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
    model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(PermaDropout(0.2))
    model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
    model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(PermaDropout(0.2))
    model.add(Dense(128))
    model.add(Dense(10,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.001, momentum=0.9),metrics=['accuracy'])
    
    return model
 
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-swdir", "--save_weights_directory", help = "Location where the weights file should be saved after training.", required=False)

    parser.add_argument("-ldwfl", "--load_weights_file", help = "Weights file that can be provided in lieu of training a new model.", required=False)

    return parser.parse_args()

def get_class_indices(class_predictions):
    zeros = [index for index in range(len(class_predictions)) if class_predictions[index] == 0]
    ones = [index for index in range(len(class_predictions)) if class_predictions[index] == 1]
    twos = [index for index in range(len(class_predictions)) if class_predictions[index] == 2]
    threes = [index for index in range(len(class_predictions)) if class_predictions[index] == 3]
    fours = [index for index in range(len(class_predictions)) if class_predictions[index] == 4]
    fives = [index for index in range(len(class_predictions)) if class_predictions[index] == 5]
    sixes = [index for index in range(len(class_predictions)) if class_predictions[index] == 6]
    sevens = [index for index in range(len(class_predictions)) if class_predictions[index] == 7]
    eights = [index for index in range(len(class_predictions)) if class_predictions[index] == 8]
    nines = [index for index in range(len(class_predictions)) if class_predictions[index] == 9]
    
    return zeros,ones,twos,threes,fours,fives,sixes,sevens,eights,nines

def get_variables_for_uncertainty_quantification(predictions_with_dropout):
    #Obtain the mean probabilities and variance of the probabilities over the ten sets of predictions. Then, make the class predictions based on the mean probabilities. The index of the greatest mean probability within each mean prediction will be designated as the predicted class, and the greatest mean probability will be called the mean predictive probability. Finally, identify the variance that corresponds to the mean predictive probability.
    
    predictive_mean = np.mean(predictions_with_dropout, axis=0)
    predictive_variance = np.var(predictions_with_dropout, axis=0)

    max_probs = np.max(predictive_mean, axis=1)
    sorted_max_probs = np.sort(np.max(predictive_mean, axis=1))
    sorted_max_probs_args = np.argsort(np.max(predictive_mean, axis=1))

    predictive_mean = [predictive_mean[i] for i in sorted_max_probs_args]
    predictive_variance = [predictive_variance[i] for i in sorted_max_probs_args]

    class_predictions = np.argmax(predictive_mean, axis=1)

    corresponding_max_vars = [None] * len(predictive_variance)

    for i in range(0, len(predictive_variance)):
         corresponding_max_vars[i] = predictive_variance[i][class_predictions[i]]
            
    return predictive_mean,predictive_variance,max_probs,sorted_max_probs,sorted_max_probs_args,class_predictions,corresponding_max_vars
    

def load_weights(weights_file):
    model = create_dropout_model()
    model.load_weights(weights_file)
    return model

def make_repeated_predictions(num_repetitions,model,x_test):
    predictions_with_dropout = [None] * num_repetitions

    for i in range(0, num_repetitions):
        predictions_with_dropout[i] = model.predict(x_test)
    
    return predictions_with_dropout

def make_predictions(model,x_test):
    predictions = model.predict(x_test)
    return predictions.argmax(axis=-1)

def plot_average_mean_predictive_probability(sorted_max_probs,legend_elements,zeros,ones,twos,threes,fours,fives,sixes,sevens,eights,nines):
    plt.figure(3,figsize=(11,6))
    plt.xlabel('Predicted Class')
    plt.ylabel('Average Mean Predictive Probability')
    plt.title('Average Mean Predictive Probability for Each Predicted Class')
    plt.scatter(0, np.mean(sorted_max_probs[zeros]), c='red')
    plt.scatter(1, np.mean(sorted_max_probs[ones]), c='pink')
    plt.scatter(2, np.mean(sorted_max_probs[twos]), c='orange')
    plt.scatter(3, np.mean(sorted_max_probs[threes]), c='brown')
    plt.scatter(4, np.mean(sorted_max_probs[fours]), c='yellow')
    plt.scatter(5, np.mean(sorted_max_probs[fives]), c='green')
    plt.scatter(6, np.mean(sorted_max_probs[sixes]), c='blue')
    plt.scatter(7, np.mean(sorted_max_probs[sevens]), c='violet')
    plt.scatter(8, np.mean(sorted_max_probs[eights]), c='gray')
    plt.scatter(9, np.mean(sorted_max_probs[nines]), c='black')
    plt.legend(handles=legend_elements)
    
    print("Pausing for 20 seconds to allow for plot observation...")
    plt.show(block=False)
    plt.pause(20)
    
    return
 
def plot_average_predictive_variance(corresponding_max_vars,legend_elements,zeros,ones,twos,threes,fours,fives,sixes,sevens,eights,nines):
    plt.figure(4,figsize=(11,6))
    plt.xlabel('Predicted Class')
    plt.ylabel('Average Predictive Variance')
    plt.title('Average Predictive Variance for Each Predicted Class')
    plt.scatter(0, np.mean(np.array(corresponding_max_vars)[zeros]), c='red')
    plt.scatter(1, np.mean(np.array(corresponding_max_vars)[ones]), c='pink')
    plt.scatter(2, np.mean(np.array(corresponding_max_vars)[twos]), c='orange')
    plt.scatter(3, np.mean(np.array(corresponding_max_vars)[threes]), c='brown')
    plt.scatter(4, np.mean(np.array(corresponding_max_vars)[fours]), c='yellow')
    plt.scatter(5, np.mean(np.array(corresponding_max_vars)[fives]), c='green')
    plt.scatter(6, np.mean(np.array(corresponding_max_vars)[sixes]), c='blue')
    plt.scatter(7, np.mean(np.array(corresponding_max_vars)[sevens]), c='violet')
    plt.scatter(8, np.mean(np.array(corresponding_max_vars)[eights]), c='gray')
    plt.scatter(9, np.mean(np.array(corresponding_max_vars)[nines]), c='black')
    plt.legend(handles=legend_elements)
    
    print("Pausing for 20 seconds to allow for plot observation...")
    plt.show(block=False)
    plt.pause(20)
    
    return
    
def plot_mean_predictive_probability(predictive_mean,sorted_max_probs,colors,legend_elements):
    plt.figure(1,figsize=(11,6))
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Predictive Probability')
    plt.title('Predicted Class and Corresponding Mean Predictive Probability for Each Test Sample (arranged in ascending order of mean predictive probability)')
    plt.scatter(range(0, len(predictive_mean)),sorted_max_probs,c=colors)
    plt.legend(handles=legend_elements)
    
    print("Pausing for 20 seconds to allow for plot observation...")
    plt.show(block=False)
    plt.pause(20)
    
    return
    
def plot_predictive_variance(predictive_variance,corresponding_max_vars,colors,legend_elements):
    plt.figure(2,figsize=(11,6))
    plt.xlabel('Sample Index')
    plt.ylabel('Predictive Variance')
    plt.title('Predicted Class and Corresponding Predictive Variance for Each Test Sample (arranged in ascending order of mean predictive probability)')
    plt.scatter(range(0, len(predictive_variance)),corresponding_max_vars,c=colors)
    plt.legend(handles=legend_elements)
    
    print("Pausing for 20 seconds to allow for plot observation...")
    plt.show(block=False)
    plt.pause(20)
    
    return

def preprocess():
    #Load and Preprocess CIFAR-10 Data

    (x_train,y_train),(x_test,y_test) = cifar10.load_data()

    #Normalize the pixel values, which are originally in the range 0 to 255, to be within 0 and 1.

    x_train = x_train/255
    x_test = x_test/255

    #Convert the labels to be categorical labels.

    y_cat_test = to_categorical(y_test,10)
    y_cat_train = to_categorical(y_train,10)
    
    return x_train,y_train,y_cat_train,x_test,y_test,y_cat_test

def train(x_train,y_cat_train):
    model = create_dropout_model()
    model.fit(x_train,y_cat_train,verbose=1,epochs=20)
    return model

#####PREPROCESSING#####

#Get data after preprocessing
x_train,y_train,y_cat_train,x_test,y_test,y_cat_test = preprocess()

#####TRAINING/LOADING/SAVING#####

#Get command-line arguments

args = get_args()

#If weights file is already provided, use the load_weights line. Otherwise, train from scratch.

if args.load_weights_file:
    print("Weights file provided. Loading model weights...")
    model = load_weights(args.load_weights_file)
else:
    print("Training model from scratch...")
    model = train(x_train,y_cat_train)
    
    #After training the model on a GPU, the final training accuracy is impressive, but some of the high accuracy value may be a result of overfitting.

#If the weights of the model are intended to be saved, save the weights file in the directory given.

if args.save_weights_directory:
    print("Save weights directory provided. Saving model weights...")
    model.save_weights(args.save_weights_directory.removesuffix('/') + '/cifar10.h5')

#####TESTING#####

#Evaluate model on test data.

print("Evaluating model...")
model.evaluate(x_test,y_cat_test)

#~78% testing accuracy is decent for a relatively simple model.

#Make predictions on the test data and get predicted class indices.

print("Making predictions and getting predicted class indices...")
predictions = make_predictions(model,x_test)

#Print classification report on model performance on test data.

print("Classification Report")
print(classification_report(y_test,predictions))

#Classes 3 and 5, which, respectively, are cat and dog, are clearly detected more poorly than the other classes. Classes 1 and 8, which, respectively, are automobile and ship, are the classes that are detected the best.

#####UNCERTAINTY QUANTIFICATION#####

#For this project, uncertainty will be quantified using the dropout layers. Dropout will be enabled during prediction, multiple predictions will be made, and the mean and variance of those predictions will be calculated to provide quantitative insight into the model's uncertainty. 

#This method is highlighted as Monte-Carlo Dropout in this article: https://medium.com/uncertainty-quantification-for-neural-networks/uncertainty-quantification-for-neural-networks-a2c5f3c1836d

#It is also described here: https://www.cs.ox.ac.uk/people/yarin.gal/website/blog_3d801aa532c1ce.html

#Replace the Dropout nodes with PermaDropout nodes so that dropout is enabled during testing. Weights will otherwise remain unchanged.

uncertainty_model = create_permadropout_model()
uncertainty_model.set_weights(model.get_weights())

#Make ten sets of predictions with dropout enabled. Each prediction thus carries some uncertainty.

print("Making ten repeated predictions...")
predictions_with_dropout = make_repeated_predictions(10,uncertainty_model,x_test)

#Get many variables for uncertainty quantification.

predictive_mean,predictive_variance,max_probs,sorted_max_probs,sorted_max_probs_args,class_predictions,corresponding_max_vars = get_variables_for_uncertainty_quantification(predictions_with_dropout)

#Create custom color map for scatter plots.

colors = list(map(color_map, class_predictions))

#Define custom legend for scatter plots based on custom color map.

legend_elements = [Line2D([0], [0], marker='o', color='red', label='Airplane',
                          markerfacecolor='red', markersize=10),
                   Line2D([0], [0], marker='o', color='pink', label='Automobile',
                          markerfacecolor='pink', markersize=10),
                   Line2D([0], [0], marker='o', color='orange', label='Bird',
                          markerfacecolor='orange', markersize=10),
                   Line2D([0], [0], marker='o', color='brown', label='Cat',
                          markerfacecolor='brown', markersize=10),
                   Line2D([0], [0], marker='o', color='yellow', label='Deer',
                          markerfacecolor='yellow', markersize=10),
                   Line2D([0], [0], marker='o', color='green', label='Dog',
                          markerfacecolor='green', markersize=10),
                   Line2D([0], [0], marker='o', color='blue', label='Frog',
                          markerfacecolor='blue', markersize=10),
                   Line2D([0], [0], marker='o', color='violet', label='Horse',
                          markerfacecolor='violet', markersize=10),
                   Line2D([0], [0], marker='o', color='gray', label='Ship',
                          markerfacecolor='gray', markersize=10),
                   Line2D([0], [0], marker='o', color='black', label='Truck',
                          markerfacecolor='black', markersize=10)]

#Plot the mean predictive probability as a function of the sample index. Because the indices are ordered such that their respective mean predictive probabilities are in ascending order, there will be an increasing curve. This plot may help in observing which predicted classes tend to have higher - and lower - mean predictive probabilities.

plot_mean_predictive_probability(predictive_mean,sorted_max_probs,colors,legend_elements)

#The part of the curve corresponding to the highest mean predictive probabilities has a "purplish" hue, which means that horse and automobile may consistently be more confidently predicted than other classes. Meanwhile, the parts of the curve corresponding to the medium and low mean predictive probabilities have a decent infusion of yellow, which means that deer and bird may be more consistently predicted with medium or low confidence than other classes.

#Similarly, plot the predictive variance as a function of the sample index. This plot may help in observing which predicted classes tend to have higher - and lower - predictive variances, especially considering that the sample indices are ordered in ascending order of mean predictive probability.

plot_predictive_variance(predictive_variance,corresponding_max_vars,colors,legend_elements)

#This plot may not at first glance be very suggestive, but the part of the plot with the lowest predictive variances - the "tail" on the right side of the plot - also has a "purplish" hue, which is consistent with the observation on the previous plot that the part of the curve with the highest mean predictive probabilities also has a "purplish" hue. Consistently high mean predictive probabilities also often result in consistently low predictive variances. Meanwhile, the middle and left side of the predictive-variance plot has a decent infusion of yellow, which coincides well with the observation on the previous plot that the parts of the curve corresponding to the medium and low mean predictive probabilities also have a decent infusion of yellow. Consistently medium and low mean predictive probabilities also often result in consistently medium and high predictive variances.

zeros,ones,twos,threes,fours,fives,sixes,sevens,eights,nines = get_class_indices(class_predictions)

#Obtain the average mean predictive probability for each predicted class, and plot it as a function of predicted class. This plot, with much fewer points, will help in clearly seeing which classes were, on average, predicted with higher - and lower - probabilities.

plot_average_mean_predictive_probability(sorted_max_probs,legend_elements,zeros,ones,twos,threes,fours,fives,sixes,sevens,eights,nines)

#The automobile class has the highest average mean predictive probability, which means that it is the class that is predicted on average with the highest confidence. The horse class is within the top five classes that are predicted on average with the highest confidence. Thus, the previous observation that the horse and automobile classes may consistently be more confidently predicted than other classes has been verified. Other classes in the top five are ship, truck, and frog. The cat class has the lowest mean average predictive probability, which means that it is the class that is predicted on average with the least confidence. Finally, the previous observation that deer and bird may be more consistently predicted with medium or low confidence than other classes has also been verified because deer and bird are in the bottom four classes that are predicted on average with the highest confidence.

#Similarly, obtain the average predictive variance for each predicted class, and plot it as a function of predicted class. This plot, also with much fewer points, will help in clearly seeing which classes were, on average, predicted with higher - and lower - variances. There may be a correspondence between the average mean predictive probability and the average predictive variance.

plot_average_predictive_variance(corresponding_max_vars,legend_elements,zeros,ones,twos,threes,fours,fives,sixes,sevens,eights,nines)

#This plot is largely consistent with the previous plot of average mean predictive probability. The classes that have the lowest average mean predictive probabilities generally also have the highest average predictive variance. In the same vein, the classes that have the highest average mean predictive probabilities generally also have the lowest average predictive variance.