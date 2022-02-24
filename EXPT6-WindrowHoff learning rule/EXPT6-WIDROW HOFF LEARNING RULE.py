#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np


# In[29]:


training_inputs = np.array([[2, 0 , 1], [2, -2, 1], [1, -2, -1]])
weights = np.array([1, 0, 1]) #np.random.rand(1,3)
d = np.array([-1, 1, 1])
print('Training Inputs:\n', training_inputs)
print('Initial weights:\n', weights)
print('d:\n', d)


# In[30]:


c = 0.5
for count in range(2):
    for i in range(len(training_inputs)):
        net = np.dot(training_inputs[i], weights)
        del_w = c * np.dot(training_inputs[i], d[i] - net)
        print(del_w)
        weights = np.add(weights, del_w, out=weights, casting="unsafe")#stackoverflowOP😍
        #weights += del_w(here weights are integers, del_w is float, different dstatype cant't be added )
    print(f'ITERATION {count + 1}:', weights)       

