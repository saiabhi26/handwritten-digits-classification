import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle
import pandas as pd

def initializeWeights(n_in, n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W

def sigmoid(z):
    return  (1/(1 + np.exp(-z)))

def preprocess():
    mat = loadmat('mnist_all.mat')

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    overall_data=np.array(np.vstack((train_data, validation_data, test_data)))
    dups = np.all(overall_data == overall_data[0,:], axis = 0)
    features_sel_data = np.where(dups==False)
    overall_data = overall_data[:,~dups]

    # Collecting selected feature indices
    features_selected = np.array([])
    features_selected = features_sel_data[0]

    train_data = overall_data[0:len(train_data),:]
    validation_data = overall_data[len(train_data): (len(train_data) + len(validation_data)),:]
    test_data = overall_data[(len(train_data) + len(validation_data)): (len(train_data) + len(validation_data) + len(test_data)),:]
    print('preprocess done')


    return train_data, train_label, validation_data, validation_label, test_data, test_label, features_selected

def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    n = train_data.shape[0]
    # Feed forward code
    w1_transpose = np.transpose(w1)
    w2_transpose = np.transpose(w2)

    input_bias = np.ones(shape=(n, 1), dtype = np.float64) # create input bias
    training_data_bias = np.append(train_data, input_bias, axis=1) # Add input bias to training data

    wij = np.dot(training_data_bias, w1_transpose) # Product of W and input data
    sig_wij = sigmoid(wij) # Sigmoid of dot product

    hidden_bias = np.ones(shape=(sig_wij.shape[0], 1), dtype = np.float64)
    biased_wij = np.append(sig_wij, hidden_bias, axis=1)

    yij = np.dot(biased_wij, w2_transpose)
    sig_yij = sigmoid(yij)

    # Labelling output
    out_ij = np.zeros(shape=(n, 10), dtype = np.float64) # setting all output values to 0 initially

    for i in range(out_ij.shape[0]):
        for j in range(out_ij.shape[1]):
            if j==training_label[i]:
                out_ij[i][j] = 1.0             #set the class labeled value to 1 and rest to 0

    # Error function

    p = out_ij*np.log(sig_yij)
    q = (1-out_ij)*np.log(1-sig_yij)
    sum1 = np.sum(p + q)
    constant = -1*n
    error = sum1/constant 

    # Regularised error function

    w1_square_sum = np.sum(np.square(w1))
    w2_square_sum = np.sum(np.square(w2))
    sum2 = w1_square_sum + w1_square_sum
    reg_factor = (sum2*lambdaval)/(2*n)
    reg_error = error + reg_factor

    obj_val = reg_error # Regularised error w.r.t lambda


    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])

    delta_l = sig_yij-out_ij
    delta_l_transpose = np.transpose(delta_l)

    grad_w2 = np.dot(delta_l_transpose, biased_wij)

    r = (1-biased_wij[:,0:n_hidden])*biased_wij[:,0:n_hidden]
    s = np.dot(delta_l, w2[:,0:n_hidden])
    rs = r*s
    rs_transpose = np.transpose(rs)
    grad_w1 = np.dot(rs_transpose, training_data_bias)

    # Regularised gradients

    reg_grad_w2 = (grad_w2 + (lambdaval*w2))/n
    reg_grad_w1 = (grad_w1 + lambdaval*w1)/n
    obj_grad = np.concatenate((reg_grad_w1.flatten(), reg_grad_w2.flatten()),0)

    return (obj_val, obj_grad)

def nnPredict(w1, w2, data):

    labels = np.zeros(data.shape[0], dtype=int)
        # Feed forward code
    input_bias = np.ones(shape=(data.shape[0],1), dtype=np.float64)
    biased_data = np.append(data, input_bias, axis=1)

    w1_transpose = np.transpose(w1)
    w2_transpose = np.transpose(w2)

    wij = np.dot(biased_data, w1_transpose)
    sig_wj = sigmoid(wij)

    hidden_bias = np.ones(shape=(sig_wj.shape[0], 1), dtype=np.float64)
    biased_wij = np.append(sig_wj, hidden_bias, axis=1)

    bl= np.dot(biased_wij, w2_transpose)
    ol = sigmoid(bl)

    for x in range(ol.shape[0]): # Label prediction
        max_arg = np.argmax(ol[x])
        labels[x] = max_arg

    return labels

train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
# n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# create a table to store the accuracy and time of every execution with no. of hidden nodes, regularization hyperparameter
results_df = pd.DataFrame(columns=['hidden nodes', 'regularization hyperparameter', 'training accuracy', 'validation accuracy', 'test accuracy', 'time taken'])

# set the regularization hyper-parameter
lambdaval = 0

n_hidden_array = [4, 8, 12, 16, 20]

for x in n_hidden_array:
    n_hidden = x
    for y in range(0, 60, 5):
        lambdaval = y

        start = time.time()

        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)

        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
        # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

        opts = {'maxiter': 50}  # Preferred value.

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

        # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        # and nnObjGradient. Check documentation for this function before you proceed.
        # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        # Test the computed parameters

        predicted_label_train = nnPredict(w1, w2, train_data)

        # print('No. of hidden units:', x)
        # print('Regularization Hyperparameter:', y)

        # find the accuracy on Training Dataset

        # print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

        predicted_label_validation = nnPredict(w1, w2, validation_data)

        # find the accuracy on Validation Dataset

        # print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

        predicted_label_test = nnPredict(w1, w2, test_data)

        # find the accuracy on Validation Dataset

        # print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

        end = time.time()

        # print('\n Time taken(in sec):'+str(end-start))

        new_row = {
            'hidden nodes': x,
            'regularization hyperparameter': y,
            'training accuracy': 100 * np.mean((predicted_label_train == train_label).astype(float)),
            'validation accuracy': 100 * np.mean((predicted_label_validation == validation_label).astype(float)),
            'test accuracy': 100 * np.mean((predicted_label_test == test_label).astype(float)),
            'time taken': end-start
        }

        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        print('-----------------------------------------------------------------------------------------------')


results_df.to_csv('results1.csv', index=False)

store_obj = dict([("selected_features", selected_features),("w1",w1), ("w2",w2), ("n_hidden",n_hidden), ("lambdaval", lambdaval)])
pickle.dump(store_obj, open('params.pickle','wb'), protocol=3)