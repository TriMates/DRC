import metrics
from loss import *
from time import time
from datasets import load_data

from IDERC import IDERC
from FcIDEC import FcIDEC

from tensorflow.keras.optimizers import SGD, Adam

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _get_data_and_model(args):

    x, y = load_data(args.dataset)
    print(x.shape)

    n_clusters = len(np.unique(y))


    if args.optimizer in ['sgd', 'SGD']:
        optimizer = SGD(args.lr, 0.9)
    else:
        optimizer = Adam()


    if 'IDERC' in args.method:
        model = IDERC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters)
        model.compile(optimizer=optimizer, loss=[self_bce, 'mse'], loss_weights=[args.rc_weight, args.rc_reco_or_weight])

    else:
        raise ValueError("Invalid value for method, which can only be in ['IDERC', 'IDERC-DA', 'IDERC-Softmax', 'IDERC-Sigmoid']")


    if '-DA' in args.method:
        args.aug_cluster = True

    return (x, y), model


def train(args):

    (x, y), model = _get_data_and_model(args)
    t0 = time()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.weights:
        model.load_weights(args.weights)
    else:
        if args.get_dec_model:

            n_clusters = len(np.unique(y))

            idec = FcIDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters)
            idec_model = idec.model
            idec_model.load_weights(args.begin_weights)

            model.autoencoder.get_layer(name='encoder_0').set_weights(idec_model.get_layer(name='encoder_0').get_weights())
            model.autoencoder.get_layer(name='encoder_1').set_weights(idec_model.get_layer(name='encoder_1').get_weights())
            model.autoencoder.get_layer(name='encoder_2').set_weights(idec_model.get_layer(name='encoder_2').get_weights())
            model.autoencoder.get_layer(name='encoder_3').set_weights(idec_model.get_layer(name='encoder_3').get_weights())
            model.autoencoder.get_layer(name='decoder_3').set_weights(idec_model.get_layer(name='decoder_3').get_weights())
            model.autoencoder.get_layer(name='decoder_2').set_weights(idec_model.get_layer(name='decoder_2').get_weights())
            model.autoencoder.get_layer(name='decoder_1').set_weights(idec_model.get_layer(name='decoder_1').get_weights())
            model.autoencoder.get_layer(name='decoder_0').set_weights(idec_model.get_layer(name='decoder_0').get_weights())

            model.model.get_layer(name='clustering').set_weights(idec_model.get_layer(name='clustering').get_weights())

        t1 = time()
        print("Time for pretraining: %ds" % (t1 - t0))


    y_pred = model.fit(x, y=y, maxiter=args.maxiter, batch_size=args.batch_size, update_interval=args.update_interval,
                       save_dir=args.save_dir, aug_cluster=args.aug_cluster)
    if y is not None:
        print('Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
              (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)))

    t2 = time()
    print("Time for pretaining, clustering and total: (%ds, %ds, %ds)" % (t1 - t0, t2 - t1, t2 - t0))
    print('='*60)

import scipy.io as scio
from sklearn.manifold import TSNE

def test(args):
    assert args.weights is not None
    fo = 3
    n_i = 2000
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=n_i)
    (x, y), model = _get_data_and_model(args)
    model.model.summary()

    print('Begin testing:', '-' * 60)
    model.load_weights(args.weights)
    y_pred = model.predict_labels(x)
    print('acc=%.4f, nmi=%.4f, ari=%.4f' % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)))
    print('End testing:', '-' * 60)
    fea = model.extract_features(x[0:2000])
    y_pred = tsne.fit_transform(fea)
    scio.savemat('drcn_fmnist_tsne.mat', {'tsne': y_pred})
    scio.savemat('drcn_fmnist_label.mat', {'label': y[0:2000]})


def tsne(args):

    assert args.weights is not None

    import matplotlib
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    from sklearn.manifold import TSNE

    (x, y), model = _get_data_and_model(args)
    model.model.summary()

    print('Begin get_tsne:', '-' * 60)
    n_i = 3000
    n_c = 10
    fo = 2

    batch_x = x[0: n_i]
    labels = y[0: n_i]
    if args.tsne_or:
        feat = batch_x
    else:
        model.load_weights(args.weights)

        feat = model.extract_features(batch_x)
    per = 30
    while per <= 50:
        per += 5
        print('tsne %d' % per)
        tsne = TSNE(perplexity=per, n_components=2, n_iter=n_i)
        tsne_enc = tsne.fit_transform(feat)

        colors = cm.rainbow(np.linspace(0, 1, n_c))

        fig, ax = plt.subplots(figsize=(3.5, 3))

        for iclass in range(0, n_c):
            idxs = labels == iclass
            ax.scatter(tsne_enc[idxs, 0], tsne_enc[idxs, 1], s=fo, c=colors[iclass], label=r'$%i$' % iclass)

        plt.xticks([-100, 0, 100], size=16)
        plt.yticks([-100, 0, 100], size=16)

        plt.tight_layout()
        plt.show()
        fig.savefig(args.save_dir + '/tsne' + str(per) + '.png', dpi=1024)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--method', default='IDERC',
                        choices=['IDERC', 'IDERC-DA', 'IDERC-Softmax', 'IDERC-Sigmoid'],
                        help="Clustering algorithm")
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'mnist-test', 'usps', 'fmnist', 'reutersidf10k'],
                        help="Dataset name to train on")
    parser.add_argument('-d', '--save-dir', default='results/temp',
                        help="Dir to save the results")

    parser.add_argument('--get-dec-model', action='store_false',
                        help="Whether to use dec model weights")
    parser.add_argument('--begin-weights', default='../DEC/results/mnist/model_final.h5', type=str,
                        help="Pretrained weights of the model")


    parser.add_argument('--get-pretrain', action='store_false',
                        help="Whether to pretrain ae")
    parser.add_argument('--pretrained-optimizer', default='sgd', type=str,
                        help="Optimizer for clustering phase")
    parser.add_argument('--pretrained-weights', default=None, type=str,
                        help="Pretrained weights of the autoencoder")
    parser.add_argument('--pretrain-epochs', default=50, type=int,
                        help="Number of epochs for pretraining")
    parser.add_argument('-v', '--verbose', default=1, type=int,
                        help="Verbose for pretraining")
    parser.add_argument('--pretrain-batch-size', default=256, type=int,
                        help="Pretrain batch size")
    parser.add_argument('--pre-lr', default=1e-3, type=float,
                        help="Weight of reco original image")
    parser.add_argument('--reco-or-weight', default=1e-4, type=float,
                        help="Weight of reco original image")
    parser.add_argument('--reco-da-weight', default=1e-4, type=float,
                        help="Weight of reco genda image")
    parser.add_argument('--pair-weight', default=1e-4, type=float,
                        help="Weight of pair loss")


    parser.add_argument('-t', '--testing', action='store',
                        help="Testing the clustering performance with provided weights")
    parser.add_argument('--get-tsneing', action='store_true',
                        help="getting tsne figure")
    parser.add_argument('-w', '--weights', default=None, type=str,
                        help="Model weights, used for testing")
    parser.add_argument('--aug-cluster', action='store_true',
                        help="Whether to use data augmentation during clustering phase")
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help="Optimizer for clustering phase")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="learning rate during clustering")
    parser.add_argument('--batch-size', default=1, type=int,
                        help="Batch size")
    parser.add_argument('--maxiter', default=2e5, type=int,
                        help="Maximum number of iterations")
    parser.add_argument('-i', '--update-interval', default=10000, type=int,
                        help="Number of iterations to update the target distribution")
    parser.add_argument('--rc-weight', default=1., type=float,
                        help="Weight of rc")

    parser.add_argument('--rc-reco-or-weight', default=1, type=float,
                        help="Weight of rc reco ori-data loss")

    args = parser.parse_args()

    import logging

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_name = args.save_dir + '/args.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(args)
    print('+' * 30, ' Parameters ', '+' * 30)
    print(args)
    print('+' * 75)


    if args.testing:
        test(args)
    elif args.get_tsneing:
        tsne(args)
    else:
        train(args)
