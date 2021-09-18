import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T   
                                   
X0 = np.concatenate((X0, np.random.randint(2, 5, N) * (np.ones((1, N)))), axis = 0)
X1 = np.concatenate((X1, np.random.randint(0, 3, N) * (np.ones((1, N)))), axis = 0) 

#X0, X1 là dữ liệu 3 chiều chứa tọa độ (x,y,z)
print ('\nx0.', X0)
print ('\nx0.', X1)


plt.plot(X0[0, :], X0[1, :], 'bs', markersize = 8, alpha = .8, label = 'positive')
plt.plot(X1[0, :], X1[1, :], 'ro', markersize = 8, alpha = .8, label = 'negative')
plt.axis('equal')
plt.legend()
plt.xlabel('X', fontsize = 20)
plt.ylabel('Y', fontsize = 20)
plt.title('Plane x,y')
plt.show()

plt.plot(X0[0, :], X0[2, :], 'bs', markersize = 8, alpha = .8, label = 'positive')
plt.plot(X1[0, :], X1[2, :], 'ro', markersize = 8, alpha = .8, label = 'negative')
plt.axis('equal')
plt.legend()
plt.xlabel('X', fontsize = 20)
plt.ylabel('Z', fontsize = 20)
plt.title('Plane x,z')
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Data 3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(X0[0, :], X0[1, :], X0[2, :], c= 'b', label = 'positive')
ax.scatter(X1[0, :], X1[1, :], X1[1, :], c ='r', label = 'negative')
plt.legend()
plt.show()

X = np.concatenate((X0, X1), axis = 1)
print ('X',X) #X = [x1,x2,y1,y2,z1,z2]
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # Tạo nhãn neg và pos
print ('\ny',y)
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0) #1 là để cho w0
print ('\nx.', X)

def h(w, x):    
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):
    return np.array_equal(h(w, X), y) #True if h(w, X) == y else False

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    mis_points = []
    e = 0
    epochs = 5 #VD data khoong lenearly seperabel thì sẽ stop sau số TG có hạn
    while True:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(4, 1) # INPUT: Vector (1,x,y)
            yi = y[0, mix_id[i]] # OUTPUT
            if h(w[-1], xi)[0] != yi: #W[-1] là input weight vector (w0,w1,w2)
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi*xi 

                w.append(w_new) # Là bảng weight của 20 data
        e = e+1
        print ('e',e)
        if has_converged(X, y, w[-1]) or e == epochs:
            break
    return (w, w_new,mis_points)

d = X.shape[0]
w_init = np.random.randn(d, 1)
#print ('winit',w_init)
(w,w_new, m) = perceptron(X, y, w_init)
print(m)
print('w',w)
print ('w_new',w_new) #Cái này thuận tiện cho việc show KQ thôi

def draw_plane(w):
    w0, w1, w2, w3 = w[0], w[1], w[2], w[3]
    XX = np.arange(2, 7, 0.1)
    YY = np.arange(-5, 5, 0.1)
    XX, YY = np.meshgrid(XX, YY)
    Z = -(w1*XX + w2*YY + w0)/w3
    #print ('Z',Z)
    return ax.plot_surface(XX, YY, Z, alpha=0.6)
    
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Data 3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(X0[0, :], X0[1, :], X0[2, :], label = 'positive')
ax.scatter(X1[0, :], X1[1, :], X1[1, :], label = 'negative')
draw_plane(w_new)
plt.legend()
plt.show()
