import numpy as np

def relu(x):
     return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    aa = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class Model_L2:
    def __init__(self, input_size, hidden_size, output_size, reg):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.reg = reg
        
        # 初始化参数
        self.W1 = np.random.randn(input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    # 前向传播
    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        y_pred = softmax(z2)
        return y_pred, a1
    
    # 计算loss
    def loss_function(self, X, y, y_pred):
        loss = -np.sum(y * np.log(y_pred))/X.shape[0]
        loss += self.reg/2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))) # L2正则化
        return loss
    
    # 反向传播
    def backward(self, X, y_true, y_pred, a1, learning_rate):
        # 输出层
        dz2 = (y_pred - y_true) / X.shape[0]
        dW2 = np.dot(a1.T, dz2) / X.shape[0] + self.reg * self.W2 / X.shape[0]
        db2 = np.sum(dz2, axis=0) / X.shape[0]
        
        # 隐藏层
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (a1 > 0)
        dW1 = np.dot(X.T, dz1) / X.shape[0] + self.reg * self.W1 / X.shape[0]
        db1 = np.sum(dz1, axis=0) / X.shape[0]

        # 参数更新
        self.W2 += -learning_rate * dW2
        self.b2 += -learning_rate * db2
        self.W1 += -learning_rate * dW1
        self.b1 += -learning_rate * db1