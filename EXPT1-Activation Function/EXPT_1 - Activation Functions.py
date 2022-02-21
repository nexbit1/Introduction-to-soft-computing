#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[26]:


#UNIPOLAR SIGNUM
array = np.arange(-5, 6, 1)
#print(array)
unipolar_signum = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
#print(unipolar_signum)
plt.step(array, unipolar_signum)
plt.show()


# In[25]:


#BIPOLAR SIGNUM
array = np.arange(-5, 6, 1)
bipolar_signum = np.array([-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
plt.step(array, bipolar_signum)
plt.show()


# In[20]:


#UNIPOLAR SIGMOID
array = np.arange(-5, 6, 1)
unipolar_sigmoid = []
for i in array:
    uni_sigmoid = 1 / (1 + np.exp(-i))
    unipolar_sigmoid.append(uni_sigmoid)
    
plt.plot(array, unipolar_sigmoid)
plt.show()


# In[21]:


#BIPOLAR SIGMOID
array = np.arange(-5, 6, 1)
bipolar_sigmoid = []
for i in array:
    bi_sigmoid = (1 - np.exp(-i)) / (1 + np.exp(-i))
    bipolar_sigmoid.append(bi_sigmoid)
plt.plot(array, bipolar_sigmoid)
plt.show()


# In[ ]:




