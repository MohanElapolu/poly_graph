import numpy as np
#from imageio import imread
#import matplotlib.pyplot as plt
from PIL import Image
#from sklearn.model_selection import train_test_split
#import numpy as np
def train_test_split1(X,Y,per,r_s):
    end = X.shape[0]
    test_size = int((end)*per)
    my_array = np.random.choice(end,test_size,replace=False)
    with open("test.txt","w+") as d_f:
        for i in range(len(my_array)):
            d_f.write("{}\n".format(my_array[i]))
    m1       = X.shape[1]
    m2       = Y.shape[1]
    train_size = end-test_size
    train_set_x = np.zeros((train_size,m1))
    test_set_x = np.zeros((test_size,m1))
    train_set_y = np.zeros((train_size,m2))
    test_set_y = np.zeros((test_size,m2))
    j = 0
    k = 0
    for i in range(end):
        if i in my_array:
            test_set_x[j,:] = X[i,:]
            test_set_y[j,:] = Y[i,:]
            j=j+1
        else:
            train_set_x[k,:] = X[i,:]
            train_set_y[k,:] = Y[i,:]
            k=k+1
    return train_set_x, test_set_x, train_set_y, test_set_y
def load_dataset(m,train_per,r_s):
    #path of the file
    image_X    = np.zeros((m,128,256,4))
    image_Y    = np.zeros((m,128,256))
   
    for i in range(1,m+1):
        path1 = "/users/melapolu/lammps/SIF_ML/path_predict/Small_panel/ptm_image/128X256/"
        pathin = path1+str(i)+"/inp.png"
        pathout = path1+str(i)+"/approx_crack.png"
        #Getting the image
        imgx = Image.open(pathin)
        image_X[i-1,:,:,:] = np.array(imgx)[:,:,:]
        imgy = Image.open(pathout)
        image_Y[i-1,:,:] = np.array(imgy)[:,:]
    #image_Y.reshape((1,-1))
    image_X_flatten = image_X.reshape((m,-1))
    image_Y_flatten = image_Y.reshape((m,-1))
    train_set_x_orig, test_set_x_orig, train_set_y_orig, test_set_y_orig  = train_test_split1(image_X_flatten,image_Y_flatten,train_per, r_s)
    classes = 2
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

