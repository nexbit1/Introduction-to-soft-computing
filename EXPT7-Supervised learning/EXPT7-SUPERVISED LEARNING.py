#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


training_inputs = np.array([[1, -2, 0, -1], [0, 1.5, -0.5, -1], [-1, 1, 0.5, -1]])
weights = np.array([1, -1, 0, 0.5])  #np.random.rand(x, no. of columns)for decimal numbers(no -ve)
d = np.array([-1, -1, 1])            #np.random.randint(-x, x, no. of colums)for integers
print('Training Inputs:\n', training_inputs)
print('Initial weights:\n', weights)
print('d:\n', d)


# In[ ]:


def bipolar_sigmoid(net):
    return round((1 - np.exp(-net)) / (1 + np.exp(-net)), 3)

def fnet_(x):#for bipolar_sigmoid          #for unipolar_sigmoid x(1-x)
    return round(((1 - x ** 2) / 2), 3)


# In[ ]:


c = 0.1
for count in range(5000): #trained 500 times
    for i in range(len(training_inputs)): #0r do enumerate or zip from iterablemodule😉
        fnet = bipolar_sigmoid(np.dot(training_inputs[i], weights))
        fnet__ = fnet_(fnet)
        del_w = c * fnet__ * np.dot(training_inputs[i],d[i]-fnet)
        weights += del_w
    print(f'ITERATION {count + 1}:', weights)       


# In[ ]:


#testing model for user input
weights = [-2.8940211 , -0.1778251, 2.81203915, 1.1965985 ]
user_input = [float(input('Enter Your Input: ')) for _ in range(4)]
input_by_user = np.array(user_input)
print(input_by_user)


# In[ ]:


output = bipolar_sigmoid(np.dot(input_by_user, weights))
print(f'OUTPUT CALCULATED: {output}')


# In[ ]:




