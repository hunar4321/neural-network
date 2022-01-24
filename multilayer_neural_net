import numpy as np
import matplotlib.pylab as plt

### some helper functions to generate sample datastes

def GenSampleDigits(start, length, order):
    chunk = ''
    i = 0
    while (len(chunk) < length):
        n = str(i + start)
        chunk += n.rjust(order, '0') + ' '
        i += 1
    return chunk

def StringToDigit(string):
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']
    if (chars[10] == string):
        dig = -1
    else:
        for j in range(len(chars) - 1):
            if (chars[j] == string):
                dig = j
                break
    return dig

def DigitToString(digit, trail):
    chars = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1]
    if (chars[10] == digit):
        d = trail
    else:
        for j in range(len(chars) - 1):
            if (chars[j] == digit):
                d = str(j)
                break
    return d

def DigitToStrings(content, trail):
    data = []
    for i in range(len(content)):
        data.append(DigitToString(content[i], trail))
    return "".join(data)

def StringToDigits(content):
    data = []
    for i in range(len(content)):
        data.append(StringToDigit(content[i]))
    return data

def act(x, atype):
    if(atype == 0):
        y=x
        dy =1
    else:
        y = np.sin(x)
        # y = 1/(1+np.exp(-x))
        dy = np.cos(x)
        # dy = y * (1-y)
    return y, dy

#######################################################
######################################################

data_type = 3
fet = 3

if(data_type == 0):
    data_len=50;
    f = open("data.txt","r" , encoding="utf8") 
    data=f.read()
    f.close()
    data = data.lower()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    idx = 0
    chunk = data[idx:idx+data_len]
    data = [stoi[s] for s in chunk]
    xs = np.asarray(data)
    x=np.zeros((fet, len(xs)))
    y=np.zeros((1, len(xs)))
    for i in range(fet):
        x[i,:] = np.roll(xs,  -i, axis=0)        
    y[0,:] = np.roll( xs, -fet, axis = 0) 
      
    X = x.T
    Y = y.T    
    # add bias term    
    ones = np.ones((x.shape[0], 1))
    x = np.hstack((x, ones))      
elif(data_type == 1):
    data_len = 50
    content = GenSampleDigits(0, data_len, 2)
    data = StringToDigits(content)      
    xs = np.asarray(data)
    x=np.zeros((fet, len(xs)))
    y=np.zeros((1, len(xs)))
    for i in range(fet):
        x[i,:] = np.roll(xs,  -i, axis=0)  
    # add bias term     
    y[0,:] = np.roll( xs, -fet, axis = 0) 
    x = x.T
    Y = y.T
    ones = np.ones((x.shape[0], 1))
    X = np.hstack((x, ones))
elif(data_type==2):
    data_len =30;
    lab=data_len;
    X = np.random.randn(lab,fet);
    bt = np.random.randn(fet,1)*0.5;
    Y = np.dot(X, bt)

elif(data_type==3):
    X=np.asarray([[0,0],[1,1],[0,1],[1,0]]);
    Y=np.asarray([[0],[0],[1],[1]]);
else:
    xs=[]
    for i in range(100):
        xs.append([i])
    X = np.asarray(xs)
    Y = x**2
    # y = 2*x


y = Y.copy()
x = X.copy()


hidden = 50
inp = x.shape[1]
out = y.shape[1]
Epochs = 1000
lr = 0.0003

W0 = np.random.randn(inp, hidden)
W1 = np.random.randn(hidden, hidden)
W2 = np.random.randn(hidden, out)

### offline version
es = []
z0 = x
for ep in range(Epochs):
    
    z1, dz1 = act(z0 @ W0, atype=1)
    z2, dz2 = act(z1 @ W1, atype=1)
    z3, dz3 = act(z2 @ W2, atype=0)
    
    e4 = y-z3
    
    e3 = (e4) * dz3 
    e2 = (e3 @ W2.T) * dz2
    e1 = (e2 @ W1.T) * dz1

    dW2 = z2.T @ e3 
    dW1 = z1.T @ e2 
    dW0 = z0.T @ e1

    W2 = W2 + dW2*lr    
    W1 = W1 + dW1*lr
    W0 = W0 + dW0*lr
    
    es.append(np.sum(np.abs(e4)))


plt.figure(1)
plt.plot(es)

z1, dz1 = act(z0 @ W0, atype=1)
z2, dz2 = act(z1 @ W1, atype=1)
z3, dz3 = act(z2 @ W2, atype=0)

for i in range(len(y)):
    print(y[i], z3[i])

plt.figure(2)
plt.plot(y)
plt.plot(z3)

