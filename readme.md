# Implementasi Feedforward Neural Network

## Deskripsi Sigkat
Repositori ini berisi implementasi algoritma *Feedforward Neural Network* (FFNN) yang dibuat from scratch menggunakan library NumPy. Model ini mendukung berbagai fungsi *activation* (Linear, ReLU, Sigmoid, Tanh, Softmax, Leaky ReLU, ELU), fungsi *loss* (MSE, BCE, CCE), serta implementasi teknik normalisasi *Root Mean Square Normalization* (RMSNorm). 

Selain implementasi model dasar, repositori ini juga berisi eksperimen untuk menguji performa model custom pada tugas klasifikasi, serta membandingkannya dengan MLP dari library Scikit-Learn

## Cara Setup dan Instalasi
Untuk menjalankan program di dalam repositori ini, pastikan telah menginstal Python di sistem. Ikuti langkah-langkah berikut untuk melakukan instalasi library yang dibutuhkan:

1. **Clone repositori ini** ke dalam penyimpanan lokal Anda:
   ```bash
   git clone https://github.com/Azekhiel/JadiMesin_Tubes1_ML.git
   cd JadiMesin_Tubes1_ML

2. **(Opsional) Buat *virtual environment*** agar pustaka tidak bentrok dengan proyek lain:
   ```bash
   python -m venv venv
   
   # Aktivasi venv (Windows)
   venv\Scripts\activate
   
   # Aktivasi venv (Linux/Mac)
   source venv/bin/activate
   ```

3. **Install semua library yang diperlukan** yang tercantum di dalam file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan Program
Eksekusi program harus dilakukan secara berurutan mulai dari prapemrosesan data hingga eksperimen evaluasi. Buka aplikasi text editor yang mendukung notebook `.ipynb`, kemudian jalankan *notebook* dengan urutan berikut:

1. **`preprocessing_classification.ipynb`**
   Buka dan jalankan seluruh *cell* pada *notebook* ini untuk melakukan prapemrosesan pada dataset klasifikasi. *Notebook* ini akan menghasilkan file dataset final dalam format untuk digunakan dalam *training*.

2. **`preprocessing_regression.ipynb`**
   Buka dan jalankan seluruh *cell* pada *notebook* ini untuk melakukan prapemrosesan pada dataset regresi. Sama seperti sebelumnya, *notebook* ini akan mengekspor data hasil pemrosesan.

3. **`experiment.ipynb`**
   Setelah kedua tahap prapemrosesan di atas selesai dan dataset *training/testing* berhasil dibuat, buka *notebook* eksperimen ini. Di sini, Anda dapat menjalankan *training* model Custom FFNN maupun Scikit-Learn, memvisualisasikan grafik *loss*, melihat distribusi bobot/gradien, serta mengevaluasi hasil akhir prediksi (*Accuracy, Confusion Matrix, MSE, R2 Score*).

## Pembagian Tugas

Proyek ini dikerjakan secara berkelompok dengan pembagian tugas sebagai berikut:

* **NIM 13222061 (William Gerald Briandelo)**: 
  * Merancang dan mengimplementasikan modul utama FFNN, termasuk kelas *Layer*, *Activation*, *Loss*, normalisasi RMSNorm, serta algoritma *forward propagation* dan *backpropagation*.
  * Penyusunan dan penulisan laporan.
* **NIM 13222033 (Anas Fathurrahman)**: 
  * Melakukan eksplorasi data dan preprocessing untuk dataset klasifikasi dan regresi.
  * Merancang dan menjalankan skenario uji pada `experiment.ipynb`.
  * Penyusunan dan penulisan laporan.