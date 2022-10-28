from tensorflow.keras.models import Model
from FcDEC import FcDEC



class FcIDEC(FcDEC):
    def __init__(self, dims, n_clusters=10, alpha=1.0):
        super(FcIDEC, self).__init__(dims, n_clusters, alpha)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.model.output, self.autoencoder.output])

    def predict(self, x):  
        q = self.model.predict(x, verbose=0)[0]
        return q

    def compile(self, optimizer='sgd', loss=['kld', 'mse'], loss_weights=[0.1, 1.0]):
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    def train_on_batch(self, x, y, sample_weight=None):
        return self.model.train_on_batch(x, [y, x], sample_weight)[0]
