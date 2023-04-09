import numpy as np

class Runner:
    def __init__(self, nn, epochs, batch_size, learning_rate):
        self.nn = nn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # 学习率衰减参数
        self.decay_rate = 0.99
        self.decay_step = 10
    
    # 训练过程
    def train(self, X, y):
        losses = []
        num_batches = int(X.shape[0] / self.batch_size)
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            self.learning_rate = self.learning_rate * (self.decay_rate ** (epoch // self.decay_step)) # 学习率指数衰减
            
            for batch in range(num_batches): # 批处理
                batch_start = batch * self.batch_size
                batch_end = (batch + 1) * self.batch_size
                X_batch = X[batch_start:batch_end, :]
                y_batch = y[batch_start:batch_end, :]
                
                y_batch_pred, a1 = self.nn.forward(X_batch) # 前向传播
                loss = self.nn.loss_function(X_batch, y_batch, y_batch_pred)
                epoch_loss += loss # loss计算
                self.nn.backward(X_batch, y_batch, y_batch_pred, a1, self.learning_rate) # 反向传播
                
            losses.append(epoch_loss/num_batches)
            if (epoch + 1) % 20 == 0:
                print('Epoch {}, loss: {}'.format(epoch+1, epoch_loss/num_batches))  
            
        return(losses)