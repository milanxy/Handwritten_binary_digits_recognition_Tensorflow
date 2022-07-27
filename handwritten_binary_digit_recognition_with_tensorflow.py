import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import BinaryCrossentropy
from keras import Sequential

import matplotlib.pylab as plt
X=np.load("Files_hdr_tensorflow/Files/X.npy")
Y=np.load("Files_hdr_tensorflow/Files/Y.npy")
print(Y.shape)
# Lets plot the data to see the intensity profiles for some random training sets
m,n=X.shape
m=1000
#fig, axes = plt.subplots(4,4)
#for i, ax in enumerate(axes.flat):
#    random_ndx =np.random.randint(m)
#    X_reshaped=X[random_ndx].reshape((20,20))
#    ax.imshow(X_reshaped, cmap='gray')




#plt.show()


#Now time has come to employ tensorflow to build, compile and create a NN to predict the numbers from their intensity profiles

X_short =X[:1000]
Y_short=Y[:1000]
print(X_short.shape)
model=Sequential([tf.keras.Input(shape=(400,)),    #specify input size
        ### START CODE HERE ### 
        Dense(units=25, activation="sigmoid"),
        Dense(units=15, activation="sigmoid"),
        Dense(units=1, activation="sigmoid")], name='my_model')
model.summary()
model.compile(loss=BinaryCrossentropy())
model.fit(X_short,Y_short,epochs=100)

#Lets predict now 
prediction = model.predict(X[0].reshape(1,400))
print(prediction, Y_short[0])

fig,axes= plt.subplots(8,8)
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]
m,n =X_short.shape
for i,ax in enumerate(axes.flat):
    random_ndx = np.random.randint(m)
    x_reshaped = X_short[random_ndx].reshape((20,20))
    prediction = model.predict(X_short[random_ndx].reshape(1,400))
    if prediction >= 0.5:
        y_hat =1
    else:
        y_hat =0
    ax.imshow(x_reshaped, cmap="gray")
    ax.set_title(f"{Y_short[random_ndx]}, {y_hat}" )
    ax.set_axis_off()
plt.show()
   
