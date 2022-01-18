import numpy as np
import matplotlib.pylab as plt

def act(x):
    y = np.sin(x)
    # y = 1/(1+np.exp(-x))
    return y
def deriv(x):
    y = np.cos(x)
    # y = x * (1-x)
    return y

x = [[1,1],[0,0],[1,0],[0,1]]
y = [[0],[0],[1],[1]]

x = np.asarray(x)
y = np.asarray(y)

hidden = 20
inp = x.shape[1]
out = y.shape[1]
Epochs = 1000
lr = 0.1

Wx = np.random.rand(inp, hidden)
Wo = np.random.rand(hidden, out)

es = []
for ep in range(Epochs):
    for i in range(y.shape[0]):
        z0 = x[i] @ Wx
        z0 = np.sin(z0)
        z1 = z0 @ Wo
        z1 = np.sin(z1)
        
        e = y[i]-z1
        es.append(e)
        
        de1 = e * deriv(z1)
        e2 = np.asarray([de1 * deriv(z0)])
        de2 = np.dot(Wx, e2.T)
        
        
        dWo = np.outer( z0 , de1 )
        dWx = np.outer( x[i], de2)
        Wo = Wo + dWo*lr
        Wx = Wx + dWx*lr

plt.figure(1)
plt.plot(es)

for i in range(y.shape[0]):
    z0 = x[i] @ Wx
    z0 = np.sin(z0)
    z1 = z0 @ Wo
    z1 = np.sin(z1)
    
    print(y[i], z1)
