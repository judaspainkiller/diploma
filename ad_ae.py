# адаптирован под изображения любого )почти) размера

import numpy as np
import tensorflow as tf
import time
import os
import signal
import sys
from datetime import timedelta
from tensorflow.keras import layers, models, losses, callbacks
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

tf.config.run_functions_eagerly(True)

# ================== НАСТРОЙКИ ==================
base_size = 64  # Базовый размер для внутренних операций
batch_size = 32  # Уменьшен из-за возможных больших изображений
epochs = 15
latent_dim = 256  # Размер латентного пространства
compression_ratio = 0.1  # Желаемый коэффициент сжатия

AE_WEIGHTS_FILE = 'adaptive_ae_weights.weights.h5'

# ================== МЕТРИКИ КАЧЕСТВА ==================
def compute_psnr(original, reconstructed, max_val=1.0):
    mse = tf.reduce_mean((original - reconstructed) ** 2)
    return 10 * tf.math.log(max_val ** 2 / mse) / tf.math.log(10.0)

def compute_ssim(original, reconstructed):
    if tf.is_tensor(original):
        original = original.numpy()
    if tf.is_tensor(reconstructed):
        reconstructed = reconstructed.numpy()
    
    if original.ndim == 3:
        return float(ssim(original, reconstructed,
                       channel_axis=-1 if original.shape[-1] > 1 else None,
                       data_range=1.0))
    return float(np.mean([ssim(original[i], reconstructed[i],
                            channel_axis=-1 if original.shape[-1] > 1 else None,
                            data_range=1.0) 
                        for i in range(original.shape[0])]))

def compute_mse(original, reconstructed):
    return tf.reduce_mean((original - reconstructed) ** 2)

# ================== ОБРАБОТЧИК СИГНАЛОВ ==================
def signal_handler(sig, frame):
    print('\nОбучение прервано! Сохраняю веса...')
    ae.save_weights(AE_WEIGHTS_FILE)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ================== ЗАГРУЗКА ДАННЫХ ==================
def preprocess_image(image, target_size=None):
    """Обработка изображения с сохранением пропорций"""
    image = tf.image.convert_image_dtype(image, tf.float32)
    if target_size:
        image = tf.image.resize(image, target_size)
    return image

def load_mnist():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = preprocess_image(np.expand_dims(x_train, -1), (base_size, base_size))
    x_test = preprocess_image(np.expand_dims(x_test, -1), (base_size, base_size))
    return x_train.numpy(), x_test.numpy()

def load_cifar10():
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_train = preprocess_image(x_train, (base_size, base_size))
    x_test = preprocess_image(x_test, (base_size, base_size))
    return x_train.numpy(), x_test.numpy()

def load_custom_image(image_path, max_dim=512):
    """Загрузка пользовательского изображения с автоматическим ресайзом"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = preprocess_image(img)
    
    # Ресайз с сохранением пропорций
    shape = tf.cast(tf.shape(img)[:2], tf.float32)
    scale = max_dim / tf.reduce_max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    return tf.image.resize(img, new_shape).numpy()

# ================== АРХИТЕКТУРА АВТОКОДИРОВЩИКА ==================
class AdaptiveAE(tf.keras.Model):
    def __init__(self, latent_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # Энкодер с адаптивным пулингом
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(32, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(64, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(128, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(latent_dim)
        ])
        
        # Декодер с фиксированным выходным размером
        self.decoder = tf.keras.Sequential([
            layers.Dense(8*8*128),
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(64, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(3, 3, activation='sigmoid', padding='same')
        ])
        
        self.metrics_trackers = {
            'loss': tf.keras.metrics.Mean(name="loss"),
            'psnr': tf.keras.metrics.Mean(name="psnr"),
            'ssim': tf.keras.metrics.Mean(name="ssim")
        }

    def call(self, inputs):
        # Автоматическое определение размера
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        
        # Масштабирование к исходному размеру
        input_shape = tf.shape(inputs)
        return tf.image.resize(decoded, (input_shape[1], input_shape[2]))

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstructions = self(data)
            loss = tf.reduce_mean(losses.mse(data, reconstructions))
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Обновление метрик
        self.metrics_trackers['loss'].update_state(loss)
        self.metrics_trackers['psnr'].update_state(compute_psnr(data, reconstructions))
        self.metrics_trackers['ssim'].update_state(compute_ssim(data, reconstructions))
        
        return {name: metric.result() for name, metric in self.metrics_trackers.items()}

# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

class TrainingMonitor(callbacks.Callback):
    def __init__(self, sample_images, vis_interval=5):
        super().__init__()
        self.sample_images = sample_images
        self.vis_interval = vis_interval
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.vis_interval == 0:
            reconstructions = self.model.predict(self.sample_images)
            plt.figure(figsize=(12, 3))
            for i in range(min(3, len(self.sample_images))):
                plt.subplot(2, 3, i+1)
                plt.imshow(self.sample_images[i])
                plt.axis('off')
                plt.subplot(2, 3, i+4)
                plt.imshow(reconstructions[i])
                plt.axis('off')
            plt.show()

def save_weights():
    try:
        ae.save_weights(AE_WEIGHTS_FILE)
        print(f"\nВеса сохранены в файл: {AE_WEIGHTS_FILE}")
    except Exception as e:
        print(f"\nОшибка сохранения весов: {str(e)}")

def load_weights():
    try:
        if os.path.exists(AE_WEIGHTS_FILE):
            ae.load_weights(AE_WEIGHTS_FILE)
            print("Веса успешно загружены из файла!")
            return True
        else:
            print("Файл весов не найден. Инициализация новых весов.")
            return False
    except Exception as e:
        print(f"Ошибка загрузки весов: {str(e)}")
        return False

# ================== CALLBACK ДЛЯ ВИЗУАЛИЗАЦИИ ==================
class AECallback(callbacks.Callback):
    def __init__(self, time_tracker, x_train, dataset_name, vis_interval=5):
        super().__init__()
        self.time_tracker = time_tracker
        self.vis_interval = vis_interval
        self.fixed_samples = x_train[:5]
        self.dataset_name = dataset_name  # Добавлено сохранение имени датасета
    
    def on_epoch_end(self, epoch, logs=None):
        current_time = format_time(time.time() - self.time_tracker.start_time)
        eta = self.time_tracker.get_eta(epoch, epochs)
        
        if epoch % self.vis_interval == 0:
            reconstructions = self.model.predict(self.fixed_samples, verbose=0)
            
            plt.figure(figsize=(12, 3))
            plt.suptitle(
                f"Автокодировщик | Датасет: {self.dataset_name}\n"
                f"Epoch {epoch+1}/{epochs} | Time: {current_time} | ETA: {eta}\n"
                f"Loss: {logs['total_loss']:.4f} | PSNR: {logs['psnr']:.2f} dB | "
                f"SSIM: {logs['ssim']:.4f} | MSE: {logs['mse']:.6f}", fontsize=10
            )
            
            for i in range(len(self.fixed_samples)):
                plt.subplot(2, len(self.fixed_samples), i+1)
                plt.imshow(self.fixed_samples[i])
                plt.axis('off')
                
                plt.subplot(2, len(self.fixed_samples), i+1+len(self.fixed_samples))
                plt.imshow(reconstructions[i])
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        print(f"\nEpoch {epoch+1}/{epochs} | Time: {current_time} | ETA: {eta}")
        print(f"Loss: {logs['total_loss']:.4f} | PSNR: {logs['psnr']:.2f} dB")
        print(f"SSIM: {logs['ssim']:.4f} | MSE: {logs['mse']:.6f}")

# ================== ОСНОВНОЙ БЛОК ==================
if __name__ == "__main__":
    # Инициализация модели
    global ae
    ae = AdaptiveAE(latent_dim=latent_dim)
    
    # Загрузка и подготовка данных
    (mnist_train, mnist_test) = load_mnist()
    (cifar_train, cifar_test) = load_cifar10()
    
    # Обучение на MNIST
    print("\n=== Обучение на MNIST ===")
    ae.compile(optimizer=Adam(1e-4))
    history = ae.fit(
        tf.data.Dataset.from_tensor_slices(mnist_train).batch(batch_size).prefetch(2),
        validation_data=tf.data.Dataset.from_tensor_slices(mnist_test).batch(batch_size),
        epochs=epochs,
        callbacks=[TrainingMonitor(mnist_train[:3])]
    )
    
    # Обучение на CIFAR-10
    print("\n=== Обучение на CIFAR-10 ===")
    history = ae.fit(
        tf.data.Dataset.from_tensor_slices(cifar_train).batch(batch_size).prefetch(2),
        validation_data=tf.data.Dataset.from_tensor_slices(cifar_test).batch(batch_size),
        epochs=epochs,
        callbacks=[TrainingMonitor(cifar_train[:3])]
    )
    
    # Тестирование на пользовательских изображениях
    test_image = load_custom_image("путь/к/изображению.jpg")
    reconstruction = ae.predict(test_image[np.newaxis, ...])[0]
    
    # Визуализация
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(test_image); plt.title("Original")
    plt.subplot(1, 2, 2); plt.imshow(reconstruction); plt.title("Reconstructed")
    plt.show()
    
    # Расчет метрик
    print(f"PSNR: {compute_psnr(test_image, reconstruction):.2f} dB")
    print(f"SSIM: {compute_ssim(test_image, reconstruction):.4f}")