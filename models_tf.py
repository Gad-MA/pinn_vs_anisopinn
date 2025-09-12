import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
from pyDOE import lhs

class Eikonal2DnetCV2:
    def __init__(self, x, y, x_e, y_e, T_e, layers, CVlayers, C=1.0, alpha=1e-5, alphaL2=1e-6):
        X = np.concatenate([x, y], 1)
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.X = X
        self.x = x
        self.y = y
        self.T_e = T_e
        self.x_e = x_e
        self.y_e = y_e
        self.layers = layers
        self.CVlayers = CVlayers


        self.lossit = []  # To track losses over time
        self.optimizer_Adam = tf.optimizers.Adam()  # Adam optimizer

        self.weights, self.biases = self.initialize_NN(layers)
        self.CVweights, self.CVbiases = self.initialize_NN(CVlayers)

        self.C = tf.constant(C, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.alphaL2 = alphaL2

        self.x_tf = tf.Variable(self.x, dtype=tf.float32)
        self.y_tf = tf.Variable(self.y, dtype=tf.float32)
        self.T_e_tf = tf.Variable(self.T_e, dtype=tf.float32)
        self.x_e_tf = tf.Variable(self.x_e, dtype=tf.float32)
        self.y_e_tf = tf.Variable(self.y_e, dtype=tf.float32)

        self.T_pred, self.CV_pred, self.f_T_pred, self.f_CV_pred = self.net_eikonal(self.x_tf, self.y_tf)
        self.T_e_pred, self.CV_e_pred, self.f_T_e_pred, self.f_CV_e_pred = self.net_eikonal(self.x_e_tf, self.y_e_tf)

        self.loss = tf.reduce_mean(tf.square(self.T_e_tf - self.T_e_pred)) + \
                    tf.reduce_mean(tf.square(self.f_T_e_pred)) + \
                    tf.reduce_mean(tf.square(self.f_T_pred)) + \
                    self.alpha * tf.reduce_mean(tf.square(self.f_CV_e_pred)) + \
                    self.alpha * tf.reduce_mean(tf.square(self.f_CV_pred)) + \
                    sum([self.alphaL2 * tf.nn.l2_loss(w) for w in self.weights])

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def initialize_NN(self, layers):
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random.normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), trainable=True)

        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), trainable=True)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.add(tf.matmul(H, W), b)
            H = tf.keras.layers.BatchNormalization()(H)  # Batch Normalization
            H = tf.tanh(H)  # Activation function
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y


    def net_eikonal(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            T = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
            CV = self.neural_net(tf.concat([x, y], 1), self.CVweights, self.CVbiases)
            CV = self.C * tf.sigmoid(CV)

        T_x = tape.gradient(T, x)
        T_y = tape.gradient(T, y)
        CV_x = tape.gradient(CV, x)
        CV_y = tape.gradient(CV, y)
        del tape

        f_T = tf.square(T_x) + tf.square(T_y) - 1.0 / tf.square(CV)
        f_CV = CV_x + CV_y

        return T, CV, f_T, f_CV


    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            loss_value = self.loss
        gradients = tape.gradient(loss_value, self.weights + self.biases)
        self.optimizer_Adam.apply_gradients(zip(gradients, self.weights + self.biases))
        return loss_value


    def compute_loss(self):
        x_tf = tf.convert_to_tensor(self.x, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(self.y, dtype=tf.float32)
        x_e_tf = tf.convert_to_tensor(self.x_e, dtype=tf.float32)
        y_e_tf = tf.convert_to_tensor(self.y_e, dtype=tf.float32)
        T_e_tf = tf.convert_to_tensor(self.T_e, dtype=tf.float32)

        T_pred, CV_pred, f_T_pred, f_CV_pred = self.net_eikonal(x_tf, y_tf)
        T_e_pred, CV_e_pred, f_T_e_pred, f_CV_e_pred = self.net_eikonal(x_e_tf, y_e_tf)

        loss = tf.reduce_mean(tf.square(T_e_tf - T_e_pred)) + \
               tf.reduce_mean(tf.square(f_T_e_pred)) + \
               tf.reduce_mean(tf.square(f_T_pred)) + \
               self.alpha * tf.reduce_mean(tf.square(f_CV_e_pred)) + \
               self.alpha * tf.reduce_mean(tf.square(f_CV_pred)) + \
               sum([self.alphaL2 * tf.nn.l2_loss(w) for w in self.weights])

        return loss


    def train(self, max_iterations=16000, patience=200):
        self.loss_history = []
        best_loss = float('inf')
        patience_counter = 0

        for it in range(max_iterations):
            with tf.GradientTape() as tape:
                loss = self.compute_loss()

            gradients = tape.gradient(loss, self.weights + self.biases + self.CVweights + self.CVbiases)
            clipped_gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]  # Gradient clipping
            self.optimizer.apply_gradients(zip(clipped_gradients, self.weights + self.biases + self.CVweights + self.CVbiases))

            self.loss_history.append(loss.numpy())

            # Print loss
            if it % 2 == 0:
                print(f"Loss: {loss.numpy():.5e}")

            # Early stopping
            if loss.numpy() < best_loss:
                best_loss = loss.numpy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter > patience:
                # print("Early stopping...")
                break


    def predict(self, x_star, y_star):
        x_star_tf = tf.convert_to_tensor(x_star, dtype=tf.float32)
        y_star_tf = tf.convert_to_tensor(y_star, dtype=tf.float32)
        T_star, CV_star, _, _ = self.net_eikonal(x_star_tf, y_star_tf)
        return T_star.numpy(), CV_star.numpy()

    def train_Adam_minibatch(self, nIter, size=50):
        """
        Train the model using mini-batches with the Adam optimizer.

        Args:
            nIter (int): Number of iterations.
            size (int): Mini-batch size.
        """
        self.lossit = []  # To track loss over iterations

        start_time = time.time()

        for it in range(nIter):
            # Generate a new mini-batch using Latin Hypercube Sampling (LHS)
            X = lhs(2, size)
            x_batch = tf.convert_to_tensor(X[:, :1], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(X[:, 1:], dtype=tf.float32)
            x_e_batch = tf.convert_to_tensor(self.x_e, dtype=tf.float32)
            y_e_batch = tf.convert_to_tensor(self.y_e, dtype=tf.float32)
            T_e_batch = tf.convert_to_tensor(self.T_e, dtype=tf.float32)

            with tf.GradientTape() as tape:
                # Compute predictions and loss
                T_pred, CV_pred, f_T_pred, f_CV_pred = self.net_eikonal(x_batch, y_batch)
                T_e_pred, CV_e_pred, f_T_e_pred, f_CV_e_pred = self.net_eikonal(x_e_batch, y_e_batch)

                loss_value = (
                    tf.reduce_mean(tf.square(T_e_batch - T_e_pred)) +
                    tf.reduce_mean(tf.square(f_T_e_pred)) +
                    tf.reduce_mean(tf.square(f_T_pred)) +
                    self.alpha * tf.reduce_mean(tf.square(f_CV_e_pred)) +
                    self.alpha * tf.reduce_mean(tf.square(f_CV_pred)) +
                    sum([self.alphaL2 * tf.nn.l2_loss(w) for w in self.weights])
                )

            # Compute gradients and apply them
            trainable_vars = self.weights + self.biases + self.CVweights + self.CVbiases
            gradients = tape.gradient(loss_value, trainable_vars)
            self.optimizer_Adam.apply_gradients(zip(gradients, trainable_vars))

            # Track loss
            self.lossit.append(loss_value.numpy())

            # Print progress every 10 iterations
            if it % 10 == 0:
                elapsed = time.time() - start_time
                C_value = np.exp(self.C.numpy())  # Exponentiate `C` to match original behavior
                print(f"It: {it}, Loss: {loss_value:.3e}, C: {C_value:.3f}, Time: {elapsed:.2f}s")
                start_time = time.time()
