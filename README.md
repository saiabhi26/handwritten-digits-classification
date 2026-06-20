Implemented a Multilayer Perceptron Neural Network and evaluated its performance in classifying handwritten digits. Used the same network to analyze a more challenging face dataset and compare the performance of the neural network. This task helps to get a better understanding of how Neural Networks work and use Feed Forward, Back Propagation to implement the Neural Networks and how regularization plays a role in the bias-variance tradeoff?

In this task :
1. Incorporated regularization on the weights (λ)
2. Used validation set to tune hyper-parameters for Neural Network (number of units in the hidden layer and λ).
3. Ran the deep neural network and compared the results with normal neural network. Ran the convolutional neural network and printed out the results like the confusion matrix. 
4. Wrote a report to further explain the experimental results.

Requirements (included in requirements.txt) :
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
tensorflow

Files included :

1. cnnScript.ipynb : This file contains code to run the convolutional neural network and also to report results (confusions matrices, test acuuracy and training time)
cnnScript.py : This file contains the same code as in cnnScript.ipynb

2. deepnnScript.ipynb : This file contains code to run the deep neural network with different number of hidden layers. It also contains the code to visualize the comparison between simple neural network and deep neural network.
deepnnScript.py : This file contains the same code as in deepnnScript.ipynb

nnScript.py : This file contains code to implement the neural network and train using the handwritten digits dataset, outputs the results into "results1.csv" and creates "params.pickle" which contains the variables required by the task.

3. results1.csv : This file contains results from training neural network on handwritten digits dataset and is used to make visualizations and to explain how to choose the optimal regularization hyperparameter.

4. params.pickle : This file contains the required variables such as list of selected features obtained after feature selection step, optimal n hidden, w1, w2 and optimal regularization coefficient. 

5. facennScript.py : This file contains the code to implement the neural network and train using the celebA dataset, outputs the results into "results2.csv".

6. results2.csv : This file contains results from training neural network on celebA dataset is used to find the best test accuracy and time taken.

7. mnist_all.mat : The MNIST dataset consists of a training set of 60000 examples and test set of 10000 examples. All digits have been size-normalized and centered in a fixed image of 28 × 28 size. In original dataset, each pixel in the image is represented by an integer between 0 and 255, where 0 is black, 255 is white and anything between represents different shade of gray. The training set of 60000 examples will be split into two sets. First set of 50000 randomly sampled examples will be used for training the neural network. The remainder 10000 examples will be used as a validation set to estimate the hyper-parameters of the network (regularization constant λ, number of hidden units).

8. face_all.pickle : sample of face images from the CelebA data set. In this file there is one data matrix and one corresponding labels vector. The preprocess routines in the script files will split the data into training and testing data

