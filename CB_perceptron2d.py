# generate data
# list of points 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 

N=10
X, y = datasets.make_blobs(n_samples=2*N, centers=2, n_features=2, center_box=(0, 15))
#Code gốc tạo dữ liệu khá khó hiểu nên mình chuyển sang code này. Nếu cần linearly seperable thì mn có thể thu nhỏ 
# center_box = (0,5)
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'ro', label = 'negative')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs', label = 'positive')
plt.axis('equal')
plt.legend()
plt.show()

X0 = X[y == 1].T # X0 = [[x],[y]]: x,y là tọa độ điểm
X1 = X[y == 0].T
print ('\nX0',X0)
print ('\nX1',X1)

X = np.concatenate((X0, X1), axis = 1)
print ('X',X) #X = [x1,x2,y1,y2]
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # Tạo nhãn cho dữ liệu neg và pos
print ('\ny',y)
# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0) #1 là để cho w0
print ('\nx.', X)

def h(w, x):    
    return np.sign(np.dot(w.T, x)) #Tính đầu ra ma trận xT và x xem neg hay pos

def has_converged(X, y, w):
    return np.array_equal(h(w, X), y) #True if h(w, X) == y else False

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    mis_points = []
    e = 0
    epochs = 5 #VD data không linearly seperabel thì sẽ stop sau số TG có hạn
    while True:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(3, 1) # INPUT: Vector (1,x,y)
            yi = y[0, mix_id[i]] # OUTPUT
            if h(w[-1], xi)[0] != yi: #W[-1] là input weight vector (w0,w1,w2)
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi*xi 
                w.append(w_new) # Là bảng weight của 20 data
                
        e = e+1
        print ('e',e)
        if has_converged(X, y, w[-1]) or e == epochs:
            break
    return (w, mis_points)
# w chỉ phục vụ cho việc lập GIF, còn lúc vẽ chỉ cần array w cuối cùng là đủ

d = X.shape[0] #Số chiều dữ liệu
w_init = np.random.randn(d, 1) #Khởi tạo w bất kì
(w, m) = perceptron(X, y, w_init)
print(m)
print(w)
# print(len(w))

def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0:
        x11, x12 = -15, 15
        return plt.plot([x11, x12], [-(w1*x11 + w0)/w2, -(w1*x12 + w0)/w2], 'k')
    else:
        x10 = -w0/w1
        return plt.plot([x10, x10], [-15, 15], 'k')

## GD example

def viz_alg_1d_2(w):
    it = len(w)    
    fig, ax = plt.subplots(figsize=(5, 5))  
    def update(i):
        plt.cla()
        #points
        plt.plot(X0[0, :], X0[1, :], 'b^', markersize = 8, alpha = .8, label = 'positive')
        plt.plot(X1[0, :], X1[1, :], 'ro', markersize = 8, alpha = .8, label = 'negative')
        #plt.axis([-5 , 17, -2, 12]) #xmin, xmax, ymin, ymax 
        plt.legend()
        if i < it:
            i2 =  i
        else: 
            i2 = it-1
        ani = draw_line(w[i2])
        if i < it-1:
            # print(X[1, m[i]], X[2, ])
            circle = plt.Circle((X[1, m[i]], X[2, m[i]]), 0.15, color='k', fill = False)
            ax.add_artist(circle)
       

        label = 'PLA: iter %d/%d' %(i2, it-1)
        ax.set_xlabel(label)
        return ani, ax 
        
    anim = FuncAnimation(fig, update, frames=np.arange(0, it + 2), interval=1000)
    anim.save('pla_vis.gif', dpi = 100, writer = 'imagemagick')
    plt.show()
#Ở Kaggle không hiện GIF mà chỉ hiện KQ cuối thôi
    
# x = np.asarray(x)
viz_alg_1d_2(w)
