Skip to content
This repository
Search
Pull requests
Issues
Gist
    @peiranzhou
        Watch 12
            Star 0
                Fork 0 Sapphirine/Analysis-of-Motor-Vehicle-Accident-in-NYC
                    Code  Issues 0  Pull requests 0  Wiki  Pulse  Graphs  Settings
Branch: master Find file Copy pathAnalysis-of-Motor-Vehicle-Accident-in-NYC/Data Processing/step3_GMM.py
c656f27  on Dec 23, 2015
@peiranzhou peiranzhou First commit
1 contributor
RawBlameHistory     230 lines (167 sloc)  5.28 KB

# coding: utf-8

# In[1]:

'''
    GMM Clustering
    input : GMM_input2.csv    6 col
    Latitude  Longitude | Month  Time  Injured  Killed
    output: GMM_output.csv   1 col
    '''
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
get_ipython().magic(u'matplotlib inline')


# In[2]:

def load_data(filename):
    '''
        ===output===
        people: list of list of int
        '''
    output = []
    file_handle = open(filename)
    for lines in file_handle:
        chunk = lines.strip().split('\r')  # list of string
        for line in chunk:
            line = line.split(',')
            #print 'line',line
            line[2] = line[2].split('/')[0] # month
            #print 'line[0]', line[0]
            line[3] = line[3].split(':')[0] # hour
            line = [float(word) for word in line]
            output.append(line)
    output = np.array(output).T
    file_handle.close()
    
    return output


# In[3]:

data = load_data("GMM_input2.csv")
location = data[0:2,:]             # for pin
data = data[2:,:]                  # for clustering
print data.shape
print location.shape


# In[4]:

# test data
data = data[:,0:10000]
location = location[:,10000]
# pre-process
norm = [12.,24.,1.1,2.1]
for i in range(2):
    data[i,:] /= norm[i]
data[2,:] += 0.1
data[2,:] /= norm[2]
data[3,:] += 0.1
data[3,:] /= norm[3]


# In[11]:

# const.
d = np.shape(data)[0] # 4
N = np.shape(data)[1] # 250

K = 4
T = 50

c = .1
alpha_prior = 1
a_prior = d
B_prior = c * d * np.cov(data)
sigma_prior = c * np.eye(d)


# In[12]:

# Initialization
phi = np.matrix(np.zeros([N,K]))
alpha = []         # list of float
m = []             # list of array
sigma = []         # list of array
a = []             # list of int
B = []             # list of array
n = [0] * K        # list of int
for j in range(K):
    alpha.append( alpha_prior )
    m.append(np.array([np.random.random(), np.random.random(), np.random.random(), np.random.random()]))
    # m.append(data[:,j*1000])
    sigma.append(sigma_prior)
    a.append( a_prior )
    B.append( B_prior )


# In[13]:

## test
i = 3000
j = 0
a[j]
term1 = np.log(sum([digamma(.5 * (1 - k + a[j])) for k in range(1,d+1)]) - np.log(np.linalg.det(B[j])))
term2 = np.log(a[j] * np.matrix(data[:,i]-m[j]) * np.linalg.inv(np.matrix(B[j])) * np.matrix(data[:,i]-m[j]).T)
term3 = np.log(np.trace(a[j] * np.linalg.inv(B[j]) * sigma[j]))
term4 = digamma(alpha[j]) - digamma(sum(alpha))
print term1, term2, term3, term4
print 'phi[i,j] =' , np.exp(.5*term1 - .5*term2[0,0] - .5*term3 + term4)


# In[14]:

# Initialization
phi = np.matrix(np.zeros([N,K]))
alpha = []         # list of float
m = []             # list of array
sigma = []         # list of array
a = []             # list of int
B = []             # list of array
n = [0] * K        # list of int

phi_list = []
z_list = []

for j in range(K):
    alpha.append( alpha_prior )
    m.append(np.array([np.random.random(), np.random.random(), np.random.random(), np.random.random()]))
    sigma.append(sigma_prior)
    a.append( a_prior )
    B.append( B_prior )
for t in range(1,T+1):
    
    # updata q(c_i)
    for i in range(N):
        for j in range(K):
            term1 = np.log(sum([digamma(.5 * (1 - k + a[j])) for k in range(1,d+1)]) - np.log(np.linalg.det(B[j])))
            term2 = np.log(a[j] * np.matrix(data[:,i]-m[j]) * np.linalg.inv(np.matrix(B[j])) * np.matrix(data[:,i]-m[j]).T)
            term3 = np.log(np.trace(a[j] * np.linalg.inv(B[j]) * sigma[j]))
            term4 = digamma(alpha[j]) - digamma(sum(alpha))
            
            
            # print term2
            
            phi[i,j] = np.exp(.5*term1 - .5*term2[0,0] - .5*term3 + term4)

    z = phi.max(axis = 1)
    phi = np.divide(phi, np.tile(z,(1,K)))

    z_list.append(z)
    phi_list.append(phi[0,2])

    for j in range(K):
        
        # set n
        n[j] = sum(phi[:,j])[0,0]
        
        # update q(pi)
        alpha[j] = alpha_prior + n[j]
        
        # update q(mu)
        sigma[j] = np.linalg.inv(sigma_prior + n[j] * a[j] * np.linalg.inv(B[j]))
        m[j] = np.array(a[j] * np.dot(sigma[j], np.dot(np.linalg.inv(B[j]),
                                                       np.matrix(sum([phi[i,j] * data[:,i] for i in range(N)])).T))).T
            
                                                       #update q(LAM)
                                                       a[j] = a_prior + n[j]
                                                       B[j] = B_prior + sum([phi[i,j] * (np.matrix(data[:,i]-m[j]).T * np.matrix(data[:,i]-m[j])
                                                                                         + sigma[j]) for i in range(N)])
## test
print "%d"%t,"%"


# In[17]:

#test
phi[50,:]
print phi_list


# In[15]:

cluster = np.argmax(phi, axis=1)    # 250 * 1 matrix
cluster_list = [cluster[i,0] for i in range(N)]
# test


# In[16]:

set(cluster_list)


# In[ ]:

plt.plot(location[0,:],location[1,:],'.')


# In[ ]:

##test
plt.figure(figsize=(9,6))
for j in range(K):
    plt.plot([location[0,i] for i in range(N) if cluster_list[i] == j],
             [location[1,i] for i in range(N) if cluster_list[i] == j],'.',label='Cluster %d'%(j+1))
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title(r"GMM for $K$ = %d"%K, fontsize=18)
plt.legend(bbox_to_anchor=(1.25,1))
plt.show()


# In[ ]:

def export_csv(filename,lst):
    myfile = open(filename, 'w')
    wr = csv.writer(myfile, delimiter='\n',quoting=csv.QUOTE_ALL)
    wr.writerow(lst)
    myfile.close()


# In[ ]:

export_csv('cluster_%d_test.csv'%K,)

Status API Training Shop Blog About
© 2016 GitHub, Inc. Terms Privacy Security Contact Help