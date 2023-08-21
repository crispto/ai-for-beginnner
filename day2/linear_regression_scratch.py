from mxnet import nd, autograd

num_samples=  100
num_input = 2

true_w = [3.8, 2.34]
true_b = 1.2

features = nd.random.normal(3, 2, shape=(num_samples, num_input))


labels = nd.dot(features, nd.array(true_w).T) + true_b


param_w = nd.random.normal(0, 1, shape=(1,2))
param_b = nd.random_normal(0,1, shape=(1,))
param_w.attach_grad()
param_b.attach_grad()

def net(X):
    return nd.dot(X, param_w.T) + param_b

def loss(pred, labels):
    return ((labels - pred.reshape(labels.shape)) ** 2).mean()


lr = 0.01
def SGD(params):
  for param in params:
     param += -1 * lr * param.grad


def get_batch(features, labels,  num_example, batch_size):
    id = nd.arange(0, num_example)
    id = nd.shuffle(id)
    for i in range(0, num_example, batch_size):
      batch = id[i:min(num_example, i+batch_size)]
      yield (features[batch], labels[batch])


def training(epoch = 20):
  for i in range(epoch):
    for feature, label in get_batch(features, labels, num_samples, 11):
        # 这里是关键，还是要靠 mxnet 提供的向后传播方法来得到梯度
        with autograd.record():
          pred = net(feature)
          l = loss(pred, label)
          l.backward()
        SGD([param_w, param_b])
      
    total_loss = loss(net(features), labels).asscalar()
    print(f"epoch: {i}, loss : {total_loss}")

print("before training", param_w, param_b)
training()

print("after training", param_w, param_b)
   
   

