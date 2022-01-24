import numpy as np
import matplotlib.pylab as plt

def act(x):
    y = np.sin(x)
    return y
def deriv(x):
    y = np.cos(x)
    return y

x = [[1,1],[0,0],[1,0],[0,1]]
y = [[0],[0],[1],[1]]

x = np.asarray(x)
y = np.asarray(y)

hidden = 5
inp = x.shape[1]
out = y.shape[1]
Epochs = 1000
lr = 0.3

Wx = np.random.rand(inp, hidden)
Wo = np.random.rand(hidden, out)

### offline version
es = []
for ep in range(Epochs):
    z1 = x @ Wx
    a1 = act(z1)
    z2 = a1 @ Wo
    a2 = act(z2)
    e1 = y-a2        
    de1 = e1 * deriv(z2)
    e2 = de1 * Wo.T
    de2 = e2 * deriv(z1)

    dWo = np.dot( z1.T , de1 )
    dWx = np.dot( x.T, de2)
    Wo = Wo + dWo*lr
    Wx = Wx + dWx*lr
    
    es.append(np.sum(np.abs(e1)))


plt.figure(1)
plt.plot(es)

z0 = x @ Wx
z0 = act(z0)
z1 = z0 @ Wo
z1 = act(z1)

for i in range(len(y)):
    print(y[i], z1[i])


# ## online version
# es = []
# for ep in range(Epochs):
#     for i in range(y.shape[0]):
#         z1 = x[i] @ Wx
#         a1 = act(z1)
#         z2 = a1 @ Wo
#         a2 = act(z2)
#         e1 = y[i]-a2        
#         de1 = e1 * deriv(z2)
#         e2 = de1 * Wo
#         de2 = e2.T * deriv(z1)

#         dWo = np.outer( z1 , de1 )
#         dWx = np.outer( x[i], de2)
#         Wo = Wo + dWo*lr
#         Wx = Wx + dWx*lr
        
#         es.append(abs(e1))


# plt.figure(1)
# plt.plot(es)

# for i in range(y.shape[0]):
#     z0 = x[i] @ Wx
#     z0 = act(z0)
#     z1 = z0 @ Wo
#     z1 = act(z1)
    
#     print(y[i], z1)
    

