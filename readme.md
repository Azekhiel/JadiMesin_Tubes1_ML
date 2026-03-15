# Buat pake ffnn-nya

```python
from ffnn import FFNN, Layer

model = FFNN(loss="bce")

# trus susun layer
# kalo mau pake RMSNorm, tambahin use_rmsnorm=True (defaultnya False)
model.add(Layer(25, 32, "relu", "he", use_rmsnorm=True))
model.add(Layer(32, 16, "relu", "he"))
model.add(Layer(16, 1, "sigmoid", "xavier"))

history = model.fit(
    X=X_train, 
    y=y_train, 
    batch_size=32, 
    epochs=50, 
    learning_rate=0.005, 
    optimizer="adam", # bisa pake optimizer lain, gd, tapi sebenernya pake else sih, jadi tulis apaan aj wkwk
    verbose=1, 
    validation_data=(X_val, y_val)
)

prediksi = model.predict(X_val)

```
## List Parameter

* **Loss Function:** mse, bce, cce
* **Aktivasi:** linear, relu, sigmoid, tanh, softmax, leaky_relu, elu
* **Inisialisasi Bobot:** zero, uniform, normal, xavier, he
* **RMSNorm:** use_rmsnorm=True atau False (defaultnya False)
* **Optimizer:** "adam" atau selain adam (gd) 
* **Regularisasi:** l1, l2 (pake desimal)
* **Buat Plotting Eksperimen:**
  * Loss per epoch: history['train_loss'] sama history['val_loss'].
  * Bobot layer: model.layers[index].weights.
  * Gradien layer: model.layers[index].d_weights.
