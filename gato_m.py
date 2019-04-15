import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import scipy
import math
from PIL import Image
from scipy import ndimage

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    print(train_set_x_orig.shape)
    
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    print("hi5"+"hola como estas")
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y, classes = load_dataset()

index = 7
plt.imshow(train_set_x_orig[index])
print("y = "+str(train_set_y_orig[:, index])+ " , es =='"+classes[np.squeeze(train_set_y_orig[:, index])].decode("utf-8")+ "'==imagen")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print(m_train)
print(m_test)
print(num_px)

train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
print(train_set_x_flatten)
plt.imshow(train_set_x_flatten)

test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
print(test_set_x_flatten)

train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255
print(train_set_x)

def initialize_with_zeros(dim):
    w= np.zeros((dim,1))
    b=0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return b,w

initialize_with_zeros(10)

def sigmoide(x):
     return 1 / (1 + math.e ** -x)

sigmoide(3.4)

def propagate(w,b,X,Y):
    m=X.shape[1]
    A=sigmoide( np.dot(w.T,X)+b)
    print(A)
    cost = -1/m * np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))
    
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * np.sum(A-Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,  "db": db}
    return grads,cost

print("salida esperada")
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs=[]
    for i in range(num_iterations):
        grads,cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w= w-dw*learning_rate
        b= b-db*learning_rate
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

def predict(w,b,X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    Y_DESTINO = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    #Obtener la predicción sin usar for
    Y_DESTINO = np.round(A)
    for i in range(A.shape[1]):
       
        if (A[0,i] <= 0.5):
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    
    assert(Y_prediction.shape == (1, m))
    print("Mi prediccion: ",Y_DESTINO)
    return Y_prediction

 
