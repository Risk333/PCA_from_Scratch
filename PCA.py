import numpy as np
import matplotlib.pyplot as plt

x_train = np.loadtxt("hist_train.csv",delimiter=",")
x_test = np.loadtxt("hist_test.csv",delimiter=',')

x_train_stand = np.empty(x_train.shape)
for i in range(x_train.shape[1]):
    x_train_stand[:,i] = (x_train[:,i]-x_train[:,i].mean())#/x_train[:,i].std()
    
x_test_stand = np.empty(x_test.shape)
for i in range(x_test.shape[1]):
    x_test_stand[:,i] = (x_test[:,i]-x_test[:,i].mean())#/x_test[:,i].std()

def transpose(mat):
    trans_mat = np.empty((mat.shape[1],mat.shape[0]))
    for i in range(mat.shape[1]):
        for j in range(mat.shape[0]):
            trans_mat[i,j] = mat[j,i]
    return trans_mat

def mal_matrix(mat1,mat2):
    malmat = np.zeros((mat1.shape[0],mat1.shape[0]))
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[0]):
            for k in range(mat1.shape[1]):
                malmat[i,j] += mat1[i,k]*mat2[k,j]
    return malmat

def find_cov_matrix(mean_mat):
    # mean_mat = (mat-mat.mean(0))
    trans_mean_mat = transpose(mean_mat)
    cov = (mal_matrix(trans_mean_mat,mean_mat))/(mean_mat.shape[0]*mean_mat.shape[0])
    return cov

def norm(arr):
    val = 0
    for i in range(arr.shape[0]):
        val+=(arr[i]*arr[i])
    return (val**(0.5))

def find_eigen_vector(dataset,n_components):
    cov_mat = find_cov_matrix(dataset)
    # eign_vals = []
    eigen_vector = np.empty((dataset.shape[1],n_components))
    for i in range(n_components):
        vector = np.random.randn(cov_mat.shape[0], 1)
        # Normalize the vector to have a norm of 1
        q = vector / norm(vector)
        for j in range(5):
            x=cov_mat@q
            q=x/norm(x)
            eigen_val = transpose(q)@(cov_mat@q)
        cov_mat = cov_mat-eigen_val*mal_matrix(q, transpose(q))
        # eign_vals.append(eigen_val)
        eigen_vector[:,i] = q[:,0]
    return eigen_vector

def projection(dataset,eigen_vector):
    projected_data=np.zeros((dataset.shape[0],2))
    for i in range(dataset.shape[0]):
        for j in range(2):
            for k in range(dataset.shape[1]):
                projected_data[i,j] += dataset[i,k]*eigen_vector[k,j]
    return projected_data

eigen_vector = find_eigen_vector(x_train_stand[:1000,:], 2)

projected_x_train = projection(x_train_stand[:1000,:],eigen_vector)
projected_x_test = projection(x_test_stand[:1000,:],eigen_vector)

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.scatter(projected_x_train[:,0],projected_x_train[:,1], color = 'red')


ax2.scatter(projected_x_test[:,0],projected_x_test[:,1], color = 'blue')