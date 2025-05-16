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

tf.config.run_functions_eagerly(True)  # Отключаем symbolic tensors

# добавить метрики на тестовую визуализацию

# ================== НАСТРОЙКИ ==================
# img_shape = (32, 32, 3)
# latent_dim = 32
img_shape = (64, 64, 3)
latent_dim = 256
batch_size = 64
epochs = 1
save_interval = 10
vis_interval = 5

# Имена файлов для весов
VAE_WEIGHTS_FILE = 'vae_compression_weights.weights.h5'
DISCRIMINATOR_WEIGHTS_FILE = 'discriminator_compression_weights.weights.h5'

# ================== МЕТРИКИ КАЧЕСТВА ==================
def compute_psnr(original, reconstructed, max_val=1.0):
    mse = tf.reduce_mean((original - reconstructed) ** 2)
    return 10 * tf.math.log(max_val ** 2 / mse) / tf.math.log(10.0)

def compute_ssim(original, reconstructed):
    """Универсальная функция вычисления SSIM для numpy и тензоров"""
    if tf.is_tensor(original):
        original = original.numpy()
    if tf.is_tensor(reconstructed):
        reconstructed = reconstructed.numpy()
    
    if isinstance(original, np.ndarray) and isinstance(reconstructed, np.ndarray):
        if original.ndim == 3:
            return float(ssim(original, reconstructed,
                           channel_axis=-1 if original.shape[-1] > 1 else None,
                           data_range=1.0))
        ssim_values = []
        for i in range(original.shape[0]):
            ssim_val = ssim(original[i], reconstructed[i],
                          channel_axis=-1 if original.shape[-1] > 1 else None,
                          data_range=1.0)
            ssim_values.append(ssim_val)
        return float(np.mean(ssim_values))
    else:
        raise ValueError("Неподдерживаемые типы данных для SSIM")

def compute_mse(original, reconstructed):
    return tf.reduce_mean((original - reconstructed) ** 2)

# ================== ОБРАБОТЧИК СИГНАЛОВ ==================
def signal_handler(sig, frame):
    print('\nОбучение прервано! Сохраняю веса...')
    save_weights()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ================== ЗАГРУЗКА ДАННЫХ MNIST ==================
def load_mnist():
    try:
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        
        # Преобразование train
        x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0
        x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train))
        x_train = tf.image.resize(x_train, img_shape[:2]).numpy()
        
        # Преобразование test
        x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0
        x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test))
        x_test = tf.image.resize(x_test, img_shape[:2]).numpy()
        
        print(f"MNIST - Train shape: {x_train.shape}, Test shape: {x_test.shape}")
        print(f"Данные проверены. Диапазон значений: {x_train.min()} - {x_train.max()}")
        
        return x_train, x_test
        
    except Exception as e:
        print(f"Ошибка загрузки MNIST: {str(e)}")
        return None, None
def load_cifar10():
    try:
        (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
        
        # Нормализация
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Ресайз до нужного размера
        x_train = tf.image.resize(x_train, img_shape[:2]).numpy()
        x_test = tf.image.resize(x_test, img_shape[:2]).numpy()
        
        print(f"\nCIFAR-10 - Train shape: {x_train.shape}, Test shape: {x_test.shape}")
        print(f"Диапазон значений: {x_train.min()} - {x_train.max()}")
        
        return x_train, x_test
        
    except Exception as e:
        print(f"Ошибка загрузки CIFAR-10: {str(e)}")
        return None, None

# ================== АРХИТЕКТУРА VAE ==================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, target_compression=0.1, min_latent=64, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.target_compression = target_compression
        self.min_latent = min_latent
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=img_shape),
            layers.Conv2D(32, 3, strides=2, activation='leaky_relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Conv2D(64, 3, strides=2, activation='leaky_relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, strides=2, activation='leaky_relu', padding='same'),  # Дополнительный слой
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, strides=2, activation='leaky_relu', padding='same'),  # 4x4x256
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(512, activation='relu')
        ])
        
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(4*4*256, activation='relu'),
            layers.Reshape((4, 4, 256)),
            layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same'),  # 32x32x32
            layers.BatchNormalization(),
            layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same'),  # 64x64x16
            layers.Conv2D(3, 3, activation='sigmoid', padding='same')
        ])
        self.encoding_times = []
        self.decoding_times = []
        self.total_encoded = 0
        self.total_decoded = 0

    def encode(self, inputs):
        start_time = time.time()
        x = self.encoder(inputs)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        # Записываем статистику
        self.encoding_times.append(time.time() - start_time)
        self.total_encoded += inputs.shape[0]  # Учитываем количество изображений в батче
  
        return z_mean, z_log_var, z
    
    def decode(self, z):
        start_time = time.time()
        decoded = self.decoder(z)
        
        # Записываем статистику
        self.decoding_times.append(time.time() - start_time)
        self.total_decoded += z.shape[0]  # Учитываем количество векторов в батче
      
        # return self.decoder(z)
        return decoded
    
    def get_stats(self):
        """Возвращает собранную статистику"""
        return {
            'total_encoded': self.total_encoded,
            'total_decoded': self.total_decoded,
            'avg_encoding_time': np.mean(self.encoding_times) if self.encoding_times else 0,
            'avg_decoding_time': np.mean(self.decoding_times) if self.decoding_times else 0,
            'last_encoding_time': self.encoding_times[-1] if self.encoding_times else 0,
            'last_decoding_time': self.decoding_times[-1] if self.decoding_times else 0
        }
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        return self.decode(z)


# ================== АРХИТЕКТУРА ДИСКРИМИНАТОРА ==================

def build_discriminator():
    return tf.keras.Sequential([
        layers.Conv2D(64, 3, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        # layers.Dropout(0.2),
        layers.Conv2D(128, 3, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        # layers.Dropout(0.2),
        # layers.Conv2D(256, 3, strides=2, padding='same'),
        layers.Flatten(),
        layers.Dense(1)  # Без sigmoid!
    ])

# ================== АРХИТЕКТУРА VAEGAN ==================
class VAEGAN(tf.keras.Model):
    def __init__(self, vae, discriminator, **kwargs):
        super(VAEGAN, self).__init__(**kwargs)
        self.vae = vae
        self.discriminator = discriminator
        
        # Явно создаем Input для feature extractor
        disc_input = layers.Input(shape=img_shape)
        disc_output = self.discriminator(disc_input)
        
        # Feature extractor из дискриминатора (берем слой перед последним)
        self.feature_extractor = tf.keras.Model(
            inputs=disc_input,
            outputs=self.discriminator.layers[-2](disc_input)  # Используем предпоследний слой
        )
        
        
        # Гиперпараметры
        self.kl_weight = 0.01
        self.recon_weight = 10.0
        self.gan_weight = 0.1
        # self.kl_weight = 0.05 очень слабое обучение
        # self.recon_weight = 1.0
        # self.gan_weight = 0.005
        
        # Метрики
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.gan_loss_tracker = tf.keras.metrics.Mean(name="gan_loss")
        self.psnr_tracker = tf.keras.metrics.Mean(name="psnr")
        self.ssim_tracker = tf.keras.metrics.Mean(name="ssim")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse")
        
        # Оптимизаторы
        # self.vae_optimizer = Adam(1e-4)
        # self.d_optimizer = Adam(4e-4)
    def call(self, inputs):
        """Основной метод для прямого прохода"""
        z_mean, z_log_var, z = self.vae.encode(inputs)
        reconstructions = self.vae.decode(z)
        return reconstructions
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.gan_loss_tracker,
            self.psnr_tracker,
            self.ssim_tracker,
            self.mse_tracker
        ]
    
    def compile(self, **kwargs):
            super(VAEGAN, self).compile(**kwargs)
            # Создаем оптимизаторы здесь
            self.vae_optimizer = Adam(1e-4)
            self.d_optimizer = Adam(1e-4)

    def train_step(self, data):
        real_images = data
        current_batch_size = tf.shape(real_images)[0]  # Динамический размер батча

        
        # Обучение дискриминатора
        self.discriminator.trainable = True
        with tf.GradientTape() as tape:
            # Генерация изображений
            z_mean, z_log_var, z = self.vae.encode(real_images)
            fake_images = self.vae.decode(z)
            
            # Предсказания дискриминатора
            real_pred = self.discriminator(real_images)
            fake_pred = self.discriminator(fake_images)
            
            # Потери дискриминатора
            # d_real_loss = losses.binary_crossentropy(tf.ones_like(real_pred), real_pred)
            # d_fake_loss = losses.binary_crossentropy(tf.zeros_like(fake_pred), fake_pred)
            d_real_loss = tf.reduce_mean(tf.square(real_pred - 1.0))  # LSGAN
            d_fake_loss = tf.reduce_mean(tf.square(fake_pred))
            d_loss = (d_real_loss + d_fake_loss) * 0.5

                # Gradient Penalty (WGAN-GP)
            alpha = tf.random.uniform(shape=[current_batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolated = real_images * alpha + fake_images * (1 - alpha)
            
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.discriminator(interpolated)
            
            gradients = gp_tape.gradient(pred, [interpolated])[0]
            gradients = tf.reshape(gradients, [current_batch_size, -1])  # [batch_size, H*W*C]
            slopes = tf.norm(gradients, axis=1)  # Норма по последнему измерению
            gradient_penalty = 10 * tf.reduce_mean(tf.square(slopes - 1.0))
            d_loss += gradient_penalty
        
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        
        # Обучение VAE
        self.discriminator.trainable = False
        with tf.GradientTape() as tape:
            # Прямой проход через VAE
            z_mean, z_log_var, z = self.vae.encode(real_images)
            reconstructions = self.vae.decode(z)

            # Вычисление метрик качества
            psnr_value = compute_psnr(real_images, reconstructions)
            ssim_value = compute_ssim(real_images, reconstructions)
            mse_value = compute_mse(real_images, reconstructions)
            
            # KL дивергенция
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            
            # Feature matching loss
            real_features = self.feature_extractor(real_images)
            fake_features = self.feature_extractor(reconstructions)
            recon_loss = tf.reduce_mean(tf.square(real_features - fake_features))
            
            # GAN loss
            validity = self.discriminator(reconstructions)
            gan_loss = tf.reduce_mean(tf.square(validity - 1.0))  # LSGAN
            # gan_loss = losses.binary_crossentropy(tf.ones_like(validity), validity)
                        # Добавляем gradient penalty:
            # with tf.GradientTape() as gp_tape:
            #     alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
            #     interpolated = real_images * alpha + reconstructions * (1 - alpha)
            #     gp_tape.watch(interpolated)
            #     pred = self.discriminator(interpolated)

            # gradients = gp_tape.gradient(pred, [interpolated])[0]
            # gradients = tf.reshape(gradients, [batch_size, -1])  # Преобразуем в [batch_size, H*W*C]
            # gradient_penalty = 10 * tf.reduce_mean(tf.square(tf.norm(gradients, axis=1) - 1.0)**2)

            # d_loss += gradient_penalty  # Добавляем к общему loss
            # Общая потеря
            total_loss = (
                self.recon_weight * recon_loss 
                + self.kl_weight * kl_loss 
                + self.gan_weight * gan_loss
            )
        
        vae_grads = tape.gradient(total_loss, self.vae.trainable_variables)
        self.vae_optimizer.apply_gradients(zip(vae_grads, self.vae.trainable_variables))
        
        # Обновление метрик
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.gan_loss_tracker.update_state(gan_loss)
        self.psnr_tracker.update_state(psnr_value)
        self.ssim_tracker.update_state(ssim_value)
        self.mse_tracker.update_state(mse_value)
        
        return {m.name: m.result() for m in self.metrics}
# def test_performance(model, test_images, num_runs=10):
#     print("\nТестирование производительности...")
#     encoding_times = []
#     decoding_times = []
    
#     for img in test_images[:num_runs]:
#         # Тест кодирования
#         start = time.time()
#         latent = model.encode(img[np.newaxis, ...])
#         encoding_times.append(time.time() - start)
        
#         # Тест декодирования
#         start = time.time()
#         _ = model.decode(latent, img.shape[:2])
#         decoding_times.append(time.time() - start)
    
#     # Вывод статистики
#     print(f"Среднее время кодирования: {np.mean(encoding_times):.4f} ± {np.std(encoding_times):.4f} сек")
#     print(f"Среднее время декодирования: {np.mean(decoding_times):.4f} ± {np.std(decoding_times):.4f} сек")
#     print(f"Общее время на {num_runs} изображений: {np.sum(encoding_times + decoding_times):.2f} сек")
    
#     return {
#         'encoding_times': encoding_times,
#         'decoding_times': decoding_times
#     }

# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

class TimeTracker:
    def __init__(self):
        self.start_time = time.time()
        self.epoch_times = []
    
    def epoch_start(self):
        self.epoch_start_time = time.time()
    
    def epoch_end(self):
        self.epoch_times.append(time.time() - self.epoch_start_time)
    
    def get_eta(self, current_epoch, total_epochs):
        avg_time = np.mean(self.epoch_times) if self.epoch_times else 0
        remaining = total_epochs - current_epoch - 1
        return format_time(avg_time * remaining)

def save_weights():
    vae.save_weights(VAE_WEIGHTS_FILE)
    discriminator.save_weights(DISCRIMINATOR_WEIGHTS_FILE)
    print(f"\nВеса сохранены в файлы: {VAE_WEIGHTS_FILE} и {DISCRIMINATOR_WEIGHTS_FILE}")

def load_weights():
    try:
        if os.path.exists(VAE_WEIGHTS_FILE) and os.path.exists(DISCRIMINATOR_WEIGHTS_FILE):
            vae.load_weights(VAE_WEIGHTS_FILE)
            discriminator.load_weights(DISCRIMINATOR_WEIGHTS_FILE)
            print("Веса успешно загружены!")
            return True
        print("Файлы весов не найдены. Инициализация новых весов.")
        return False
    except Exception as e:
        print(f"Ошибка загрузки весов: {str(e)}")
        return False

# ================== CALLBACK ДЛЯ ВИЗУАЛИЗАЦИИ ==================
class VAEGANCallback(callbacks.Callback):
    def __init__(self, time_tracker, x_train, dataset_name, vis_interval=5):
        super().__init__()
        self.time_tracker = time_tracker
        self.vis_interval = vis_interval
        self.fixed_samples = x_train[:5]
        self.dataset_name = dataset_name
    
    def on_epoch_end(self, epoch, logs=None):
        current_time = format_time(time.time() - self.time_tracker.start_time)
        eta = self.time_tracker.get_eta(epoch, epochs)
        
        if epoch % self.vis_interval == 0:
            reconstructions = self.model.vae.predict(self.fixed_samples, verbose=0)
            
            plt.figure(figsize=(12, 3))
            plt.suptitle(
                f"VAEGAN | Датасет: {self.dataset_name}\n"
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
        print(f"Total Loss: {logs['total_loss']:.4f} | Recon: {logs['recon_loss']:.4f}")
        print(f"KL Loss: {logs['kl_loss']:.4f} | GAN Loss: {logs['gan_loss']:.4f}")
        print(f"PSNR: {logs['psnr']:.2f} dB | SSIM: {logs['ssim']:.4f} | MSE: {logs['mse']:.6f}")

# ================== ОСНОВНОЙ БЛОК ==================
if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Загрузка данных
    x_train, x_test = load_mnist()
    if x_train is None:
        sys.exit(1)
    
    # Инициализация моделей
    vae = VAE()
    discriminator = build_discriminator()
    
    # Явное построение моделей
    vae.build(input_shape=(None, *img_shape))
    discriminator.build(input_shape=(None, *img_shape))
    
    # Проверка архитектуры
    test_input = tf.random.uniform((1, *img_shape))
    test_output = vae(test_input)
    print(f"Тест архитектуры VAE пройден. Вход: {test_input.shape}, Выход: {test_output.shape}")
    
    test_output = discriminator(test_input)
    print(f"Тест архитектуры Discriminator пройден. Вход: {test_input.shape}, Выход: {test_output.shape}")

    # Загрузка весов
    load_weights()
    
    # Инициализация VAEGAN
    vae_gan = VAEGAN(vae, discriminator)
    
    # Подготовка callback'ов
    time_tracker = TimeTracker()
    callback = VAEGANCallback(time_tracker, x_train, vis_interval)
    
    # Компиляция модели
    vae_gan.compile()
    # Список датасетов для обучения
    datasets = [
        ("MNIST", load_mnist),
        ("CIFAR-10", load_cifar10)
    ]
    # Обучение на каждом датасете
    for dataset_name, load_func in datasets:
        print(f"\n=== Начало обучения на {dataset_name} ===")
    
        # Загрузка данных
        x_train, x_test = load_func()
        if x_train is None:
            continue

        # Подготовка callback'ов
        time_tracker = TimeTracker()
        callback = VAEGANCallback(time_tracker, x_train, dataset_name)

        # Обучение
        try:
            train_ds = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            # print("\n=== Начало обучения VAEGAN ===")
            history = vae_gan.fit(
                train_ds,
                epochs=epochs,
                callbacks=[callback],
                verbose=1
            )
            save_weights()

            # Тестирование после обучения на датасете
            print(f"\n=== Тестирование на {dataset_name} ===")
            test_samples = x_test[:100]
            reconstructions = vae_gan.predict(test_samples, verbose=0)
            if tf.is_tensor(test_samples):
                test_samples_np = test_samples.numpy()
            else:
                test_samples_np = test_samples
                
            if tf.is_tensor(reconstructions):
                reconstructions_np = reconstructions.numpy()
            else:
                reconstructions_np = reconstructions
                
            psnr = float(compute_psnr(test_samples, reconstructions).numpy())
            ssim_val = compute_ssim(test_samples_np, reconstructions_np)
            mse = float(compute_mse(test_samples, reconstructions).numpy())
                
            print(f"\nРезультаты на тестовых данных ({dataset_name}):")
            print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f} | MSE: {mse:.6f}")
                    # Добавьте в тестовый блок:
            test_img = x_test[0:1]  # Берем одно тестовое изображение
            test_img_tensor = tf.convert_to_tensor(test_img)  # Конвертируем в тензор
            z_mean, z_log_var, z = vae.encode(test_img_tensor)
            # Получаем размеры в байтах
            original_size = test_img.nbytes  # Для numpy array используем .nbytes
            compressed_size = z.numpy().nbytes  # z - это тензор, для него используем .numpy()
            print(f"Исходный размер: {original_size} байт")
            print(f"Сжатое представление (z): {compressed_size} байт")
            print(f"Коэффициент сжатия: {original_size / compressed_size:.1f}x")    
        # Дополнительная статистика
           # Получаем статистику
            stats = vae.get_stats()

            print("\nДетальная статистика:")
            print(f"Всего операций кодирования: {stats['total_encoded']}")
            print(f"Среднее время кодирования: {stats['avg_encoding_time']:.4f} сек")
            print(f"Всего операций декодирования: {stats['total_decoded']}")
            print(f"Среднее время декодирования: {stats['avg_decoding_time']:.4f} сек")
            print(f"Последнее время кодирования: {stats['last_encoding_time']:.4f} сек")
            print(f"Последнее время декодирования: {stats['last_decoding_time']:.4f} сек")   
            # Визуализация тестовых примеров
            plt.figure(figsize=(10, 4))
            plt.suptitle(
                    f"VAEGAN\n"
                    # f"Epoch {epoch+1}/{epochs} | Time: {current_time} | ETA: {eta}\n"
                    # f"Loss: {logs['total_loss']:.4f} | PSNR: {logs['psnr']:.2f} dB | "
                    # f"SSIM: {logs['ssim']:.4f} | MSE: {logs['mse']:.6f}", fontsize=10
                )
            plt.suptitle(f"Тестовые примеры ({dataset_name})", fontsize=12)
            for i in range(3):
                plt.subplot(2, 3, i+1)
                plt.imshow(test_samples[i])
                plt.title("Original")
                plt.axis('off')
                    
                plt.subplot(2, 3, i+4)
                plt.imshow(reconstructions[i])
                plt.title("Reconstructed")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
        
        except Exception as e:
            print(f"\nОшибка обучения: {str(e)}")
        finally:
            total_time = format_time(time.time() - time_tracker.start_time)
            print(f"\nОбщее время выполнения: {total_time}")
            
