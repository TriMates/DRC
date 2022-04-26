import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec, Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import VarianceScaling

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)

    return K.sum(x * y, axis=-1, keepdims=True)


def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0])

def build_encoder(dims, act='relu'):
    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')

    # input
    x = Input(shape=(dims[0],), name='encoder_input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)

    return Model(inputs=x, outputs=h, name='encoder')


def build_decoder(dims, act='relu'):
    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')

    x = Input(shape=(dims[-1],), name='encoder_input')

    y = x
    # internal layers in decoder
    for i in range(n_stacks - 1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='decoder')


def build_dualae(dims):
    base_encoder = build_encoder(dims)
    base_decoder = build_decoder(dims)

    enc_in_shape = base_encoder.input_shape[1:]

    enc_in_a = Input(shape=enc_in_shape, name='input_a')
    enc_in_b = Input(shape=enc_in_shape, name='input_b')

    feat_a = base_encoder(enc_in_a)
    feat_b = base_encoder(enc_in_b)

    y_a = base_decoder(feat_a)
    y_b = base_decoder(feat_b)

    z = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([feat_a, feat_b])

    return Model(inputs=[enc_in_a, enc_in_b], outputs=[y_a, y_b, z])

class RCLayer(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(RCLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape.as_list()[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.sigmoid(q)

        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], )

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(RCLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RCLayer_sigmoid_2(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(RCLayer_sigmoid_2, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape.as_list()[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.sigmoid(q)
        q = 2. * q - 1.

        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], )

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(RCLayer_sigmoid_2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RCLayer_softmax(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(RCLayer_softmax, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape.as_list()[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.softmax(q, axis=1)

        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], )

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(RCLayer_softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def autoencoder(dims, act='relu'):

    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')

    # input
    in_a = Input(shape=(dims[0],), name='input_a')
    in_b = Input(shape=(dims[0],), name='input_b')

    h_a = in_a
    h_b = in_b

    # internal layers in encoder
    for i in range(n_stacks-1):
        dense = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)
        h_a = dense(h_a)
        h_b = dense(h_b)

    # hidden layer
    dense = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))
    h_a = dense(h_a)
    h_b = dense(h_b)

    z = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([h_a, h_b])

    y_a = h_a
    y_b = h_b

    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        dense = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)
        y_a = dense(y_a)
        y_b = dense(y_b)

    # output
    dense = Dense(dims[0], kernel_initializer=init, name='decoder_0')
    y_a = dense(y_a)
    y_b = dense(y_b)

    return Model(inputs=[in_a, in_b], outputs=[y_a, y_b, z], name='AE'), \
           Model(inputs=in_a, outputs=h_a, name='encoder')
