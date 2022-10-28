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

    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')


    x = Input(shape=(dims[0],), name='input')
    h = x


    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)


    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  

    y = h

    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)


    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)


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


        rc_layer = RCLayer(self.n_clusters, name='clustering')(self.encoder.output)


        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[rc_layer, self.autoencoder.output], name='derc')  

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


        t0 = time()
        if not aug_pretrain:
            self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb, verbose=verbose)
        else:
            print('-=*' * 20)
            print('Using augmentation for ae')
            print('-=*' * 20)

            def gen(x, batch_size):
                if len(x.shape) > 2: 
                    gen0 = self.datagen.flow(x, shuffle=True, batch_size=batch_size)
                    while True:
                        batch_x = gen0.next()
                        yield (batch_x, batch_x)
                else:
                    width = int(np.sqrt(x.shape[-1]))
                    if width * width == x.shape[-1]:  
                        im_shape = [-1, width, width, 1]
                    else:  
                        width = int(np.sqrt(x.shape[-1] / 3.0))
                        im_shape = [-1, width, width, 3]
                    gen0 = self.datagen.flow(np.reshape(x, im_shape), shuffle=True, batch_size=batch_size)
                    while True:
                        batch_x = gen0.next()
                        batch_x = np.reshape(batch_x, [batch_x.shape[0], x.shape[-1]])
                        yield (batch_x, batch_x)

            self.autoencoder.fit_generator(gen(x, batch_size), steps_per_epoch=int(x.shape[0] / batch_size),
                                           epochs=epochs, callbacks=cb, verbose=verbose)

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

        q = self.model.predict(x, verbose=0)[0]

        return q

    def predict_labels(self, x): 

        q = self.model.predict(x, verbose=0)[0]
        return np.argmax(q, 1)

    @staticmethod
    def target_distribution(q): 
        qmax = q.max(1)
        valid = np.ones(1)
        fake = np.zeros(1)

        seed = np.random.rand(1)
        out = np.where(qmax > seed, valid, fake)

        return out

    def random_transform(self, x):
        if len(x.shape) > 2:  
            return self.datagen.flow(x, shuffle=False, batch_size=x.shape[0]).next()


        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]: 
            im_shape = [-1, width, width, 1]
        else:  
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
        save_interval = 7e4 
        print('Save interval', save_interval)
        t1 = time()


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
                q = self.predict(x)  
                p = self.target_distribution(q)  

                y_pred1 = q.argmax(1)


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


            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/model_' + str(ite) + '.h5')



            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]


            loss += self.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            ite += 1


        logfile.close()
        print('saving model to:', save_dir + '/model_final.h5')
        self.model.save_weights(save_dir + '/model_final.h5')
        print('Clustering time: %ds' % (time() - t1))
        print('End clustering:', '-' * 60)

        return y_pred1
