'''
    采用units中的RCLayer的call函数部分 使用柯西分布 + sigmoid
    IDERC_sigmoid_2.py 采用的是RCLayer_sigmoid_2 柯西分布 + sigmoid + 2 * fx - 1 fx指sigmoid计算后的结果
    IDERC_softmax.py 采用的是RCLayer_softmax 柯西分布 +softmax

    三个分布都是基于IDERC中定义的IDERC框架的，不同的部分只有强化聚类层

    target_distribution函数是经过修改过的，对应于计算y值的过程
    loss.py 对应于得到奖赏后计算损失的过程，是选择最大的预测概率计算损失后调用框架中的train_on_batch进行训练的

    用的上的部分就是从三个分布中选择一个使用->RCLayer RCLayer_sigmoid_2 RCLayer_softmax
    计算目标分布y->target_distribution
    对最大概率分配奖赏并计算损失->loss.py

'''
from loss import *
from units import *
from time import time

from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import metrics
from sklearn.cluster import KMeans

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')

    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)
    # y = K.softmax(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class IDERC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0):

        super(IDERC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrained = False
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)

        self.autoencoder, self.encoder = autoencoder(self.dims)

        # prepare DERC model
        rc_layer = RCLayer(self.n_clusters, name='clustering')(self.encoder.output)

        # prepare FcDEC model
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[rc_layer, self.autoencoder.output], name='derc')   #  todo

    def pretrain(self, x, y=None, x_test=None, y_test=None, optimizer='adam', epochs=200, batch_size=256,
                 save_dir='results/temp', verbose=1, aug_pretrain=False):
        print('Begin pretraining: ', '-' * 60)
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        best_acc = 0.80
        if y_test is not None or y is not None and verbose > 0:

            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs / 10) != 0 and epoch % int(epochs / 100) != 0:
                        return

                    feature_model = Model(self.model.input,
                                          self.model.get_layer(index=int(len(self.model.layers) / 2)).output)
                    print(self.x.shape)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    ACC = metrics.acc(self.y, y_pred)
                    NMI = metrics.nmi(self.y, y_pred)
                    if ACC > best_acc:
                        self.model.save_weights((save_dir + '/ae_weights_' + str(ACC) + '.h5'))

                    print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (ACC, NMI))

            if y_test is not None:
                cb.append(PrintACC(x_test, y_test))
            else:
                cb.append(PrintACC(x[0: int(x.shape[0] / 2)], y[0: int(x.shape[0] / 2)]))

        # begin pretraining
        t0 = time()
        if not aug_pretrain:
            self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb, verbose=verbose)
        else:
            print('-=*' * 20)
            print('Using augmentation for ae')
            print('-=*' * 20)

            def gen(x, batch_size):
                if len(x.shape) > 2:  # image
                    gen0 = self.datagen.flow(x, shuffle=True, batch_size=batch_size)
                    while True:
                        batch_x = gen0.next()
                        yield (batch_x, batch_x)
                else:
                    width = int(np.sqrt(x.shape[-1]))
                    if width * width == x.shape[-1]:  # gray
                        im_shape = [-1, width, width, 1]
                    else:  # RGB
                        width = int(np.sqrt(x.shape[-1] / 3.0))
                        im_shape = [-1, width, width, 3]
                    gen0 = self.datagen.flow(np.reshape(x, im_shape), shuffle=True, batch_size=batch_size)
                    while True:
                        batch_x = gen0.next()
                        batch_x = np.reshape(batch_x, [batch_x.shape[0], x.shape[-1]])
                        yield (batch_x, batch_x)

            self.autoencoder.fit_generator(gen(x, batch_size), steps_per_epoch=int(x.shape[0] / batch_size),
                                           epochs=epochs, callbacks=cb, verbose=verbose)
            # self.autoencoder.fit_generator(gen(x, batch_size), steps_per_epoch=int(x.shape[0]/batch_size),
            #                                epochs=epochs, callbacks=cb, verbose=verbose,
            #                                workers=8, use_multiprocessing=True if platform.system() != "Windows" else False)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True
        print('End pretraining: ', '-' * 60)

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        # print(x.shape)
        q = self.model.predict(x, verbose=0)[0]

        return q

    def predict_labels(self, x):  # predict cluster labels using the output of clustering layer

        q = self.model.predict(x, verbose=0)[0]
        return np.argmax(q, 1)

    @staticmethod
    def target_distribution(q):   #  todo
        qmax = q.max(1)
        valid = np.ones(1)
        fake = np.zeros(1)

        seed = np.random.rand(1)
        out = np.where(qmax > seed, valid, fake)

        return out

    def random_transform(self, x):
        if len(x.shape) > 2:  # image
            return self.datagen.flow(x, shuffle=False, batch_size=x.shape[0]).next()

        # if input a flattened vector, reshape to image before transform
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen = self.datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=x.shape[0])
        return np.reshape(gen.next(), x.shape)

    def compile(self, optimizer='sgd', loss=[self_bce, 'mse'], loss_weights=[0.1, 1.]):
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    def train_on_batch(self, x, y, sample_weight=None):
        return self.model.train_on_batch(x, [y, x], sample_weight)[0]

    def fit(self, x, y=None, maxiter=7e6, batch_size=256,
            update_interval=140, save_dir='./results/temp', aug_cluster=False):

        self.model.summary()
        print('Begin clustering:', '-' * 60)
        print('Update interval', update_interval)
        save_interval = 7e4  # only save the initial and final model
        print('Save interval', save_interval)
        t1 = time()

        # Step 2: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.predict(x)  # 柯西分布+sigmoid todo
                p = self.target_distribution(q)  # y = 0&1 todo

                y_pred1 = q.argmax(1)

                # evaluate the clustering performance
                avg_loss = loss / update_interval
                loss = 0.
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred1), 5)
                    nmi = np.round(metrics.nmi(y, y_pred1), 5)
                    ari = np.round(metrics.ari(y, y_pred1), 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=avg_loss)
                    logwriter.writerow(logdict)
                    logfile.flush()
                    print('Iter %d: acc=%.5f, nmi=%.5f, ari=%.5f; loss=%.5f' % (ite, acc, nmi, ari, avg_loss))

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/model_' + str(ite) + '.h5')


            # train on batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            # x_batch = self.random_transform(x[idx]) if aug_cluster else x[idx]

            loss += self.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/model_final.h5')
        self.model.save_weights(save_dir + '/model_final.h5')
        print('Clustering time: %ds' % (time() - t1))
        print('End clustering:', '-' * 60)

        return y_pred1
