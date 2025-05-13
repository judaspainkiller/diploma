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

# ================== НАСТРОЙКИ ==================
img_shape = (32, 32, 3)
batch_size = 64
epochs = 20
save_interval = 10
vis_interval = 5

AE_WEIGHTS_FILE = 'ae_weights.weights.h5'

# ================== МЕТРИКИ КАЧЕСТВА ==================
def compute_psnr(original, reconstructed, max_val=1.0):
    mse = tf.reduce_mean((original - reconstructed) ** 2)
    return 10 * tf.math.log(max_val ** 2 / mse) / tf.math.log(10.0)

def compute_ssim(original, reconstructed):
    """Универсальная функция вычисления SSIM для numpy и тензоров"""
    # Преобразуем в numpy массивы если нужно
    if tf.is_tensor(original):
        original = original.numpy()
    if tf.is_tensor(reconstructed):
        reconstructed = reconstructed.numpy()
    
    # Если уже numpy массивы
    if isinstance(original, np.ndarray) and isinstance(reconstructed, np.ndarray):
        # Для одного изображения
        if original.ndim == 3:
            return float(ssim(original, reconstructed,
                           channel_axis=-1 if original.shape[-1] > 1 else None,
                           data_range=1.0))
        # Для батча изображений
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
    ae.save_weights(AE_WEIGHTS_FILE)
    print(f"Веса сохранены в {AE_WEIGHTS_FILE}")
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
    
# ================== АРХИТЕКТУРА АВТОКОДИРОВЩИКА ==================
class AE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=img_shape),
            layers.Conv2D(32, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
        ])
        
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(3, 3, activation='sigmoid', padding='same')
        ])
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.psnr_tracker = tf.keras.metrics.Mean(name="psnr")
        self.ssim_tracker = tf.keras.metrics.Mean(name="ssim")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.psnr_tracker, 
                self.ssim_tracker, self.mse_tracker]

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstructions = self(data)
            recon_loss = tf.reduce_mean(losses.mse(data, reconstructions))
            total_loss = recon_loss
        
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        psnr_value = compute_psnr(data, reconstructions)
        ssim_value = compute_ssim(data, reconstructions)
        mse_value = compute_mse(data, reconstructions)
        
        self.total_loss_tracker.update_state(total_loss)
        self.psnr_tracker.update_state(psnr_value)
        self.ssim_tracker.update_state(ssim_value)
        self.mse_tracker.update_state(mse_value)
        
        return {m.name: m.result() for m in self.metrics}

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

# def load_custom_images(image_dir, target_size=(32, 32)):
#     image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))]
#     images = []
#     for path in image_paths:
#         img = tf.keras.utils.load_img(path, target_size=target_size)
#         img = tf.keras.utils.img_to_array(img) / 255.0
#         images.append(img)
#     return np.array(images)

# # Использование:
# custom_images = load_custom_images("path/to/your/images")
# def test_on_custom_images(model, image_dir):
#     images = load_custom_images(image_dir)
#     if len(images) == 0:
#         print("Нет изображений в директории!")
#         return
    
#     reconstructions = model.predict(images)
    
#     # Вычисление метрик
#     psnr = compute_psnr(images, reconstructions).numpy()
#     ssim_val = compute_ssim(images, reconstructions)
#     mse = compute_mse(images, reconstructions).numpy()
    
#     print("\nРезультаты на пользовательских изображениях:")
#     print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f} | MSE: {mse:.6f}")
    
#     # Визуализация
#     plt.figure(figsize=(10, 4))
#     for i in range(min(3, len(images))):
#         plt.subplot(2, 3, i+1)
#         plt.imshow(images[i])
#         plt.title("Original")
#         plt.axis('off')
        
#         plt.subplot(2, 3, i+4)
#         plt.imshow(reconstructions[i])
#         plt.title("Reconstructed")
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

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
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Загрузка данных
    x_train, x_test = load_mnist()
    if x_train is None:
        sys.exit(1)
    
    # Инициализация модели
    global ae  # Делаем модель глобальной для signal_handler
    ae = AE()
    ae.build(input_shape=(None, *img_shape))
    
    # Проверка архитектуры
    test_input = tf.random.uniform((1, *img_shape))
    test_output = ae(test_input)
    print(f"Тест архитектуры пройден. Вход: {test_input.shape}, Выход: {test_output.shape}")

    # Загрузка весов
    load_weights()
    
    # Подготовка callback'ов
    time_tracker = TimeTracker()
    callback = AECallback(time_tracker, x_train, vis_interval)
    
    # Компиляция модели
    ae.compile(optimizer=Adam(1e-4))
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
        callback = AECallback(time_tracker, x_train, dataset_name)

        # Обучение
        try:
            train_ds = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            # print("\n=== Начало обучения ===")
            history = ae.fit(
                train_ds,
                epochs=epochs,
                callbacks=[callback],
                verbose=1
            )
            save_weights()
        
        # Тестирование после обучения на датасете
            print(f"\n=== Тестирование на {dataset_name} ===")
            test_samples = x_test[:100]
            reconstructions = ae.predict(test_samples, verbose=0)

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
            
            # Визуализация тестовых примеров
            plt.figure(figsize=(10, 4))
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
            print(f"\nОшибка обучения на {dataset_name}: {str(e)}")
        
        finally:
            total_time = format_time(time.time() - time_tracker.start_time)
            print(f"\nОбщее время обучения на {dataset_name}: {total_time}")

    # test_on_custom_images(ae, "path/to/your/images")        
            # # Тестирование
            # if x_test is not None:
            #     print("\n=== Финальное тестирование ===")
            #     test_samples = x_test[:100]
            #     reconstructions = ae.predict(test_samples, verbose=0)

            #             # Убедимся, что данные в правильном формате
            #     if tf.is_tensor(test_samples):
            #         test_samples_np = test_samples.numpy()
            #     else:
            #         test_samples_np = test_samples
                
            #     if tf.is_tensor(reconstructions):
            #         reconstructions_np = reconstructions.numpy()
            #     else:
            #         reconstructions_np = reconstructions
                
            #     psnr = float(compute_psnr(test_samples, reconstructions).numpy())
            #     ssim_val = compute_ssim(test_samples_np, reconstructions_np)
            #     mse = float(compute_mse(test_samples, reconstructions).numpy())
                
            #     print(f"\nРезультаты на тестовых данных:")
            #     print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f} | MSE: {mse:.6f}")
                
            #     # Визуализация
            #     plt.figure(figsize=(10, 4))
            #     plt.suptitle("Тестовые примеры", fontsize=12)
            #     for i in range(3):
            #         plt.subplot(2, 3, i+1)
            #         plt.imshow(test_samples[i])
            #         plt.title("Original")
            #         plt.axis('off')
                    
            #         plt.subplot(2, 3, i+4)
            #         plt.imshow(reconstructions[i])
            #         plt.title("Reconstructed")
            #         plt.axis('off')
            #     plt.tight_layout()
            #     plt.show()