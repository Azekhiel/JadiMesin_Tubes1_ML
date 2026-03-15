import numpy as np

class Activation:
    @staticmethod
    def linear(x): return x
    @staticmethod
    def linear_deriv(x): return np.ones_like(x)

    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def relu_deriv(x): return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x): 
        x = np.clip(x, -500, 500) # biar ga overflow pas di exp
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def sigmoid_deriv(x):
        sig = Activation.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def tanh(x): return np.tanh(x)
    @staticmethod
    def tanh_deriv(x): return 1 - np.tanh(x)**2

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # bonus: 2 fungsi aktivasi tambahan
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    @staticmethod
    def leaky_relu_deriv(x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    @staticmethod
    def elu_deriv(x, alpha=1.0):
        return np.where(x > 0, 1.0, Activation.elu(x, alpha) + alpha)


class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    @staticmethod
    def mse_deriv(y_true, y_pred):
        n = y_true.shape[0]
        return -(2/n) * (y_true - y_pred)

    @staticmethod
    def bce(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        n = y_true.shape[0]
        return -(1/n) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    @staticmethod
    def bce_deriv(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        n = y_true.shape[0]
        return -(1/n) * ((y_true / y_pred) - ((1 - y_true) / (1 - y_pred)))

    @staticmethod
    def cce(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        n = y_true.shape[0]
        return -(1/n) * np.sum(y_true * np.log(y_pred))
    @staticmethod
    def cce_deriv(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        n = y_true.shape[0]
        return -(1/n) * (y_true / y_pred)

class RMSNorm:
    def __init__(self, size, eps=1e-8):
        self.eps = eps
        self.gamma = np.ones((1, size))
        
        # cache untuk backprop
        self.x = None
        self.rms = None
        self.x_norm = None
        
        # gradien
        self.d_gamma = None
        
        # adam memory untuk gamma
        self.m_gamma = np.zeros_like(self.gamma)
        self.v_gamma = np.zeros_like(self.gamma)

    def forward(self, x):
        self.x = x
        # hitung Root Mean Square sepanjang dimensi fitur (axis=1)
        self.rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True) + self.eps)
        self.x_norm = x / self.rms
        return self.gamma * self.x_norm

    def backward(self, d_out):
        N, D = d_out.shape
        
        # gradien untuk gamma
        self.d_gamma = np.sum(d_out * self.x_norm, axis=0, keepdims=True)
        
        # gradien untuk input x
        d_x_norm = d_out * self.gamma
        
        # turunan RMSNorm
        d_rms = np.sum(d_x_norm * self.x * (-1.0 / (self.rms**2)), axis=1, keepdims=True)
        d_x = (d_x_norm / self.rms) + (d_rms * self.x / (D * self.rms))
        
        return d_x

class Layer:
    def __init__(self, input_size, output_size, activation="relu", init_method="uniform", seed=None, use_rmsnorm=False, **kwargs):
        if seed is not None:
            np.random.seed(seed)
            
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.use_rmsnorm = use_rmsnorm 
        
        # map fungsi aktivasi
        self.activations = {
            "linear": (Activation.linear, Activation.linear_deriv),
            "relu": (Activation.relu, Activation.relu_deriv),
            "sigmoid": (Activation.sigmoid, Activation.sigmoid_deriv),
            "tanh": (Activation.tanh, Activation.tanh_deriv),
            "softmax": (Activation.softmax, None),
            "leaky_relu": (Activation.leaky_relu, Activation.leaky_relu_deriv),
            "elu": (Activation.elu, Activation.elu_deriv)
        }
        self.act_func, self.act_deriv = self.activations[self.activation_name]
        
        # inisialisasi bobot
        if init_method == "zero":
            self.weights = np.zeros((input_size, output_size))
        elif init_method == "uniform":
            low = kwargs.get('lower_bound', -1.0)
            high = kwargs.get('upper_bound', 1.0)
            self.weights = np.random.uniform(low, high, (input_size, output_size))
        elif init_method == "normal":
            mean = kwargs.get('mean', 0.0)
            var = kwargs.get('variance', 1.0)
            self.weights = np.random.normal(mean, np.sqrt(var), (input_size, output_size))
        elif init_method == "xavier":
            # bagus buat sigmoid/tanh
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        elif init_method == "he":
            # bagus buat relu dkk
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * 0.01 # fallback
            
        self.bias = np.zeros((1, output_size))
        
        self.rmsnorm = RMSNorm(output_size) if self.use_rmsnorm else None

        # cache buat backprop
        self.inputs = None
        self.z = None
        self.z_pre_act = None
        self.d_weights = None
        self.d_bias = None

        # variabel memori buat adam optimizer
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias

        if self.use_rmsnorm:
            self.z_pre_act = self.rmsnorm.forward(self.z)
        else:
            self.z_pre_act = self.z
            
        return self.act_func(self.z_pre_act)

    def backward(self, d_out):
        if self.activation_name != "softmax":
            d_z_pre_act = d_out * self.act_deriv(self.z_pre_act)
        else:
            # d_out dari softmax+cce udah gabungan
            d_z_pre_act = d_out
        
        if self.use_rmsnorm:
            d_z = self.rmsnorm.backward(d_z_pre_act)
        else:
            d_z = d_z_pre_act
            
        self.d_weights = np.dot(self.inputs.T, d_z)
        self.d_bias = np.sum(d_z, axis=0, keepdims=True)
        
        # return gradien buat dilanjutin ke layer sebelumnya
        d_inputs = np.dot(d_z, self.weights.T)
        return d_inputs


class FFNN:
    def __init__(self, loss="mse"):
        self.layers = []
        self.losses = {
            "mse": (Loss.mse, Loss.mse_deriv),
            "bce": (Loss.bce, Loss.bce_deriv),
            "cce": (Loss.cce, Loss.cce_deriv)
        }
        self.loss_func, self.loss_deriv = self.losses[loss.lower()]
        self.loss_name = loss.lower()

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def fit(self, X, y, batch_size=32, epochs=100, learning_rate=0.01, verbose=1, 
            optimizer="adam", l1=0.0, l2=0.0, validation_data = None):
        
        history = {'train_loss': [], 'val_loss':[]}
        n_samples = X.shape[0]
        
        # hyperparameter adam
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        t = 0 # timestep buat adam

        for epoch in range(epochs):
            # ngacak data tiap epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            num_batches = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # forward
                out = self.predict(X_batch)
                epoch_loss += self.loss_func(y_batch, out)
                num_batches += 1
                
                # backward
                if self.loss_name == "cce" and self.layers[-1].activation_name == "softmax":
                    d_out = (out - y_batch) / X_batch.shape[0]
                else:
                    d_out = self.loss_deriv(y_batch, out)

                for layer in reversed(self.layers):
                    d_out = layer.backward(d_out)
                    
                # update bobot
                t += 1
                for layer in self.layers:
                    # hitung penalti l1 l2
                    l1_pen = l1 * np.sign(layer.weights)
                    l2_pen = l2 * layer.weights
                    
                    grad_w = layer.d_weights + l1_pen + l2_pen
                    grad_b = layer.d_bias
                    
                    if optimizer == "adam":
                        # adam optimizer
                        layer.m_w = beta1 * layer.m_w + (1 - beta1) * grad_w
                        layer.v_w = beta2 * layer.v_w + (1 - beta2) * (grad_w ** 2)
                        
                        layer.m_b = beta1 * layer.m_b + (1 - beta1) * grad_b
                        layer.v_b = beta2 * layer.v_b + (1 - beta2) * (grad_b ** 2)
                        
                        m_w_hat = layer.m_w / (1 - beta1 ** t)
                        v_w_hat = layer.v_w / (1 - beta2 ** t)
                        
                        m_b_hat = layer.m_b / (1 - beta1 ** t)
                        v_b_hat = layer.v_b / (1 - beta2 ** t)
                        
                        layer.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                        layer.bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
                        
                        if layer.use_rmsnorm:
                            layer.rmsnorm.m_gamma = beta1 * layer.rmsnorm.m_gamma + (1 - beta1) * layer.rmsnorm.d_gamma
                            layer.rmsnorm.v_gamma = beta2 * layer.rmsnorm.v_gamma + (1 - beta2) * (layer.rmsnorm.d_gamma ** 2)
                            m_g_hat = layer.rmsnorm.m_gamma / (1 - beta1 ** t)
                            v_g_hat = layer.rmsnorm.v_gamma / (1 - beta2 ** t)
                            layer.rmsnorm.gamma -= learning_rate * m_g_hat / (np.sqrt(v_g_hat) + epsilon)
                    else:
                        # gradient descent
                        layer.weights -= learning_rate * grad_w
                        layer.bias -= learning_rate * grad_b
                        
                        if layer.use_rmsnorm:
                            layer.rmsnorm.gamma -= learning_rate * layer.rmsnorm.d_gamma

            avg_train_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_train_loss)

            if validation_data is not None:
                X_val, y_val = validation_data
                val_out = self.predict(X_val)
                val_loss = self.loss_func(y_val, val_out)
                history['val_loss'].append(val_loss)
            
            if verbose == 1:
                if validation_data is not None:
                    print(f"epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}")

        return history