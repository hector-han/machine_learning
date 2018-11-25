import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


const_A = 0.1
def run_training(train_X, train_Y):

    # weights
    b = tf.Variable(tf.zeros([], dtype=tf.float64))
    W = tf.Variable(tf.random_uniform([2,1], -0.1, 0.1,  dtype=tf.float64))

    m=len(train_X)
    # linear model
    y = tf.matmul(train_X, W) + b
    cost = tf.reduce_sum(tf.square(y - train_Y)) / (2*m)
    gradient = tf.train.GradientDescentOptimizer(const_A)
    optimizer = gradient.minimize(cost)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(20000):

            sess.run(optimizer)

            if step % 2000 == 0:
                print("Step:{}, Cost={}, W={}, b={}".format(step+1, sess.run(cost), sess.run(W),sess.run(b)))

        print("Optimization Finished!")
        training_cost = sess.run(cost)
        print("Training Cost={}, W={}, b={}".format(training_cost, sess.run(W), sess.run(b)))

def read_data(filename):
    with open(filename) as fd:
        data_list = fd.read().splitlines()

        m = len(data_list) # number of examples
        n = 2 # number of features

        train_X = np.zeros([m, n], dtype=np.float64)
        train_Y = np.zeros([m, 1], dtype=np.float64)

        for i in range(m):
            datas = data_list[i].split(",")
            for j in range(n):
                train_X[i][j] = float(datas[j])
            train_Y[i][0] = float(datas[-1])
    return train_X, train_Y

def fake_data():
    x_data = np.float32(np.random.rand(2, 100)) # 随机输入
    x_data = np.transpose(x_data)
    y_data = np.dot(x_data, [[0.100], [0.200]]) + 0.300
    return x_data, y_data

def feature_normalize(train_X):
    train_X_tmp = train_X.transpose()
    for N in range(2):
        train_X_tmp[N] = train_X_tmp[N] / np.max((train_X_tmp[N]))
    train_X = train_X_tmp.transpose()

    return train_X

def cal_fuction_val(X, y, omega):
    t = np.matmul(X, omega) - y
    return np.sum(t**2)


def cal_gradient(X, y, omega):
    XT = np.transpose(X)
    return np.matmul(np.matmul(XT, X), omega) - np.matmul(XT, y)


def my_solve(train_X, train_y):
    m = len(train_X)
    ones = np.ones([m, 1], dtype=np.float64)
    X = np.concatenate((ones, train_X), axis = 1)
    design = np.matmul(np.transpose(X), X)
    w,v = np.linalg.eig(design)
    analysis = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), train_Y))
    analysis_val = cal_fuction_val(X, train_Y, analysis)/2/m
    print('w={}, v={}'.format(w,v))
    print('analysis: val={} with omega={}'.format(analysis_val, analysis))

    print(design)
    omega = np.zeros([3,1], dtype=np.float64)
    const_A = 0.0001
    alpha = const_A
    v = cal_fuction_val(X, train_Y, omega)
    for step in range(20000):
        omega_1 = omega - (alpha/m) * cal_gradient(X, train_Y, omega)
        v_1 = cal_fuction_val(X, train_Y, omega_1)
        if v_1 < v:
            alpha = const_A
            omega = omega_1
            v = v_1
        else:
            alpha = alpha/2
        if step % 2000 == 0:
            print('cost={}, omega={}'.format(v/2/m, omega))



if __name__ == '__main__':
    filename = 'train.dat'
    train_X, train_Y = read_data(filename)

    train_X = feature_normalize(train_X)
    print('=================tensorflow output==========')
    run_training(train_X, train_Y)
    print('=================my_solve output==========')
    my_solve(train_X, train_Y)

