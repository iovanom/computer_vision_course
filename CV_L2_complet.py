# %% [markdown]
# # Laborator 2 — Segmentarea semantică a imaginilor cu U-Net
#
# **Plan:**
# - **Partea A** (sarcinile 1–3): setul de date CARLA (mașini autonome, 23 de clase), U-Net construit pas cu pas, antrenare, evaluare.
# - **Partea B** (sarcina 4): un set de date nou — **Oxford-IIIT Pet** (disponibil și pe Kaggle) — colectare, măști, preprocesare cu OpenCV, împărțire 80/10/10, structura `Dataset/`, antrenare U-Net, evaluare pe test.
#
# **Mediu:** micromamba `cv`, Linux, CPU. TensorFlow 2.21 + `tf-keras` (Keras 2), activat prin `TF_USE_LEGACY_KERAS=1` pentru compatibilitate cu `test_utils.py` (care folosește `layer.output_shape`, inexistent în Keras 3).

# %% [markdown]
# ## 0 - Configurare mediu
# Variabilele de mediu trebuie setate **înainte** de primul `import tensorflow`.

# %%
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # tf.keras -> tf_keras (Keras 2)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # mai puține mesaje de log de la TF

import tensorflow as tf
import numpy as np
import cv2

print("TensorFlow:", tf.__version__)
print("Keras folosit de tf.keras:", tf.keras.__name__)
print("NumPy:", np.__version__, "| OpenCV:", cv2.__version__)
print("Dispozitive:", tf.config.list_physical_devices())

tf.keras.utils.set_random_seed(42)  # reproductibilitate

# %% [markdown]
# ## Introducere
# Segmentarea semantică atribuie **o clasă fiecărui pixel**. Spre deosebire de detecția de obiecte (bounding boxes),
# obținem o mască precisă pentru fiecare obiect.
#
# <img src="https://raw.githubusercontent.com/ahsan-83/Deep-Learning-Specialization-Coursera/eed436df9cea96bc5999da0b27c0cc2cb27caa54/Convolutional%20Neural%20Networks/Assignment/Week%203-Programming%20Assignment%20Image%20Segmentation%20with%20U-Net/images/carseg.png" style="width:500px;">
#
# <center><b>Figura 1</b>: Exemplu de imagine segmentată</center>

# %% [markdown]
# <a name='1'></a>
# ## 1 - Pachete

# %%
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

from test_utils import summary, comparator

import pandas as pd
import imageio.v2 as imageio          # imageio.imread (v3) e depreciat; folosim API-ul v2
import matplotlib.pyplot as plt
from pathlib import Path

# %% [markdown]
# <a name='2'></a>
# ## 2 - Încărcarea și studierea datelor (CARLA)
#
# **Corectură față de original:** `os.listdir()` NU garantează ordinea. Dacă listele de imagini și măști nu sunt sortate,
# perechile (imagine, mască) pot fi greșite → modelul învață pe etichete greșite. De aceea sortăm și verificăm.

# %%
path = ''
image_path = os.path.join(path, './data/CameraRGB/')
mask_path = os.path.join(path, './data/CameraMask/')
assert os.path.isdir(image_path) and os.path.isdir(mask_path), "Pune setul de date în ./data/CameraRGB și ./data/CameraMask"

image_list = sorted(os.listdir(image_path))
mask_list = sorted(os.listdir(mask_path))
assert image_list == mask_list, "Numele fișierelor din CameraRGB și CameraMask nu corespund!"

image_list = [image_path + i for i in image_list]
mask_list = [mask_path + i for i in mask_list]
print("Număr imagini:", len(image_list), "| Număr măști:", len(mask_list))

# %% [markdown]
# ### Studiul setului de date (sarcina 2)
# Verificăm dimensiunile, tipul datelor și ce valori (clase) apar în măști.
# În CARLA, eticheta clasei se află în **canalul roșu** al măștii (de aceea vom lua `max` pe canale).

# %%
img0 = imageio.imread(image_list[0]); m0 = imageio.imread(mask_list[0])
print("Imagine:", img0.shape, img0.dtype, "| Mască:", m0.shape, m0.dtype)

classes_seen = set()
for p in mask_list[::max(1, len(mask_list)//60)]:      # ~60 măști eșantionate
    m = imageio.imread(p)
    classes_seen |= set(np.unique(m[..., 0]).tolist())
print("Clase observate (canal R):", sorted(classes_seen), "-> total", len(classes_seen))

# %%
N = 2
img = imageio.imread(image_list[N])
mask = imageio.imread(mask_list[N])

fig, arr = plt.subplots(1, 2, figsize=(14, 10))
arr[0].imshow(img);            arr[0].set_title('Image')
arr[1].imshow(mask[:, :, 0]);  arr[1].set_title('Segmentation')
plt.show()

# %% [markdown]
# <a name='2-1'></a>
# ### 2.1 - Setul de date ca `tf.data.Dataset` (perechi imagine–mască)

# %%
image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)
for p in zip(image_list_ds.take(3), mask_list_ds.take(3)):
    print(p)

# %%
image_filenames = tf.constant(image_list)
masks_filenames = tf.constant(mask_list)
dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
for image, mask in dataset.take(1):
    print(image); print(mask)

# %% [markdown]
# <a name='2-2'></a>
# ### 2.2 - Preprocesarea datelor
#
# - `process_path`: citește PNG-ul, convertește imaginea în `float32` **în intervalul [0,1]** (`convert_image_dtype` face deja împărțirea la 255);
#   masca devine un singur canal (max pe canale = clasa).
# - `preprocess`: redimensionare la 96×128 cu `nearest` (pentru mască e obligatoriu `nearest` — nu vrem clase „interpolate”).
#
# **Corectură față de original:** originalul mai împărțea o dată la 255 în `preprocess`, deci pixelii ajungeau în [0, 0.004].
# Am eliminat a doua împărțire.

# %%
def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)   # -> [0,1]

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')
    return input_image, input_mask

image_ds = dataset.map(process_path)
processed_image_ds = image_ds.map(preprocess)

# %% [markdown]
# <a name='3'></a>
# ## 3 - U-Net
#
# U-Net (2015, inițial pentru imagini medicale) extinde FCN: în loc de o singură convoluție transpusă la final,
# are un **encoder** (reduce rezoluția, crește canalele) și un **decoder** simetric (crește rezoluția cu `Conv2DTranspose`),
# plus **conexiuni skip** care aduc detaliile fine din encoder în decoder.
#
# <img src="https://raw.githubusercontent.com/ahsan-83/Deep-Learning-Specialization-Coursera/eed436df9cea96bc5999da0b27c0cc2cb27caa54/Convolutional%20Neural%20Networks/Assignment/Week%203-Programming%20Assignment%20Image%20Segmentation%20with%20U-Net/images/unet.png" style="width:700px;">
#
# <center><b>Figura 2</b>: Arhitectura U-Net</center>
#
# Noi folosim `padding='same'`, deci nu e nevoie de crop înainte de concatenare (dimensiunile coincid).

# %% [markdown]
# <a name='3-2'></a>
# ### 3.2 - Encoder: Exercițiul 1 — `conv_block`
# 2 × Conv2D(3×3, ReLU, he_normal, same) → [Dropout] → [MaxPooling 2×2].
# Returnează `(next_layer, skip_connection)`; skip-ul e ieșirea **înainte** de pooling.
#
# <img src="https://raw.githubusercontent.com/ahsan-83/Deep-Learning-Specialization-Coursera/eed436df9cea96bc5999da0b27c0cc2cb27caa54/Convolutional%20Neural%20Networks/Assignment/Week%203-Programming%20Assignment%20Image%20Segmentation%20with%20U-Net/images/encoder.png" style="width:500px;">

# %%
def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """Bloc convoluțional de reducere a dimensiunii. Returnează next_layer, skip_connection."""
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    if dropout_prob > 0:
        conv = Dropout(rate=dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPooling2D(2, 2)(conv)
    else:
        next_layer = conv

    skip_connection = conv
    return next_layer, skip_connection

# %%
input_size = (96, 128, 3)
n_filters = 32
inputs = Input(input_size)
cblock1 = conv_block(inputs, n_filters * 1)
model1 = tf.keras.Model(inputs=inputs, outputs=cblock1)
output1 = [['InputLayer', [(None, 96, 128, 3)], 0],
           ['Conv2D', (None, 96, 128, 32), 896, 'same', 'relu', 'HeNormal'],
           ['Conv2D', (None, 96, 128, 32), 9248, 'same', 'relu', 'HeNormal'],
           ['MaxPooling2D', (None, 48, 64, 32), 0, (2, 2)]]
print('Block 1:')
for layer in summary(model1):
    print(layer)
comparator(summary(model1), output1)

inputs = Input(input_size)
cblock1 = conv_block(inputs, n_filters * 32, dropout_prob=0.1, max_pooling=True)
model2 = tf.keras.Model(inputs=inputs, outputs=cblock1)
output2 = [['InputLayer', [(None, 96, 128, 3)], 0],
           ['Conv2D', (None, 96, 128, 1024), 28672, 'same', 'relu', 'HeNormal'],
           ['Conv2D', (None, 96, 128, 1024), 9438208, 'same', 'relu', 'HeNormal'],
           ['Dropout', (None, 96, 128, 1024), 0, 0.1],
           ['MaxPooling2D', (None, 48, 64, 1024), 0, (2, 2)]]
print('\nBlock 2:')
for layer in summary(model2):
    print(layer)
comparator(summary(model2), output2)

# %% [markdown]
# <a name='3-3'></a>
# ### 3.3 - Decoder: Exercițiul 2 — `upsampling_block`
# `Conv2DTranspose(n_filters, 3, strides=2, same)` dublează H și W → concatenare `[up, contractive_input]` pe axa 3
# (conexiunea skip) → 2 × Conv2D ca în encoder.
#
# <img src="https://raw.githubusercontent.com/ahsan-83/Deep-Learning-Specialization-Coursera/eed436df9cea96bc5999da0b27c0cc2cb27caa54/Convolutional%20Neural%20Networks/Assignment/Week%203-Programming%20Assignment%20Image%20Segmentation%20with%20U-Net/images/decoder.png" style="width:500px;">

# %%
def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """Bloc convoluțional de creștere a dimensiunii."""
    up = Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    return conv

# %%
input_size1 = (12, 16, 256)
input_size2 = (24, 32, 128)
n_filters = 32
expansive_inputs = Input(input_size1)
contractive_inputs = Input(input_size2)
cblock1 = upsampling_block(expansive_inputs, contractive_inputs, n_filters * 1)
model1 = tf.keras.Model(inputs=[expansive_inputs, contractive_inputs], outputs=cblock1)
output1 = [['InputLayer', [(None, 12, 16, 256)], 0],
           ['Conv2DTranspose', (None, 24, 32, 32), 73760],
           ['InputLayer', [(None, 24, 32, 128)], 0],
           ['Concatenate', (None, 24, 32, 160), 0],
           ['Conv2D', (None, 24, 32, 32), 46112, 'same', 'relu', 'HeNormal'],
           ['Conv2D', (None, 24, 32, 32), 9248, 'same', 'relu', 'HeNormal']]
print('Block 1:')
for layer in summary(model1):
    print(layer)
comparator(summary(model1), output1)

# %% [markdown]
# <a name='3-4'></a>
# ### 3.4 - Exercițiul 3 — `unet_model`
# Encoder: filtre 32→64→128→256→512, dropout 0.3 la ultimele două blocuri, ultimul fără pooling (bottleneck).
# Decoder: 256→128→64→32, cu skip-urile `cblockX[1]`. Final: Conv2D 3×3 + Conv2D 1×1 cu `n_classes` filtre (logits).

# %%
def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    inputs = Input(input_size)
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters * 2)
    cblock3 = conv_block(cblock2[0], n_filters * 4)
    cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    return tf.keras.Model(inputs=inputs, outputs=conv10)

# %%
import outputs
img_height, img_width, num_channels = 96, 128, 3
unet = unet_model((img_height, img_width, num_channels))
comparator(summary(unet), outputs.unet_model_output)

# %% [markdown]
# <a name='3-5'></a>
# ### 3.5 - Dimensiunile modelului și rezumatul

# %%
unet = unet_model((img_height, img_width, num_channels))
unet.summary()

# %% [markdown]
# <a name='3-6'></a>
# ### 3.6 - Funcția de pierdere
# Măștile conțin indexul clasei (0..22), nu vectori one-hot → **SparseCategoricalCrossentropy**.
# `from_logits=True` pentru că ultimul strat nu are softmax.

# %%
unet.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

# %% [markdown]
# <a name='3-7'></a>
# ### 3.7 - Afișarea datelor

# %%
def display(display_list, titles=('Input Image', 'True Mask', 'Predicted Mask')):
    plt.figure(figsize=(15, 15))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# %%
for image, mask in image_ds.take(1):
    sample_image, sample_mask = image, mask
    print(mask.shape)
display([sample_image, sample_mask])

# %%
for image, mask in processed_image_ds.take(1):
    sample_image, sample_mask = image, mask
    print(mask.shape)
display([sample_image, sample_mask])

# %% [markdown]
# <a name='4'></a>
# ## 4 - Antrenarea modelului
# Pe CPU, 40 de epoci pot dura mult (zeci de minute, depinde de procesor).
# Pentru un test rapid, pune `QUICK_RUN = True` (5 epoci).

# %%
QUICK_RUN = False
EPOCHS = 5 if QUICK_RUN else 40
BUFFER_SIZE = 500
BATCH_SIZE = 32

train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print(processed_image_ds.element_spec)
model_history = unet.fit(train_dataset, epochs=EPOCHS)

os.makedirs("models", exist_ok=True)
unet.save("models/carla_unet.keras")

# %% [markdown]
# <a name='4-1'></a>
# ### 4.1 - Măștile prezise
# `argmax` pe axa claselor → clasa cu scorul maxim pentru fiecare pixel.

# %%
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# %% [markdown]
# <a name='4-2'></a>
# ### 4.2 - Acuratețea și pierderea pe parcursul antrenării

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(model_history.history["accuracy"]); ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoca")
ax[1].plot(model_history.history["loss"]);     ax[1].set_title("Loss");     ax[1].set_xlabel("epoca")
plt.show()

# %% [markdown]
# ### Metrici de segmentare: Pixel Accuracy și mIoU
# Acuratețea pe pixeli poate fi înșelătoare (clasele mari domină). **IoU** per clasă = TP / (TP + FP + FN);
# **mIoU** = media IoU pe clasele prezente. Le calculăm dintr-o matrice de confuzie.
# (Notă: aici evaluăm pe datele de antrenare, ca în laboratorul original; în Partea B evaluăm corect pe test.)

# %%
def evaluate_segmentation(model, ds, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for x, y in ds:
        p = np.argmax(model.predict(x, verbose=0), axis=-1).astype(np.int64).ravel()
        t = y.numpy().astype(np.int64).ravel()
        cm += np.bincount(t * n_classes + p, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    pixel_acc = tp.sum() / cm.sum()
    return pixel_acc, iou, np.nanmean(iou)

pa, iou, miou = evaluate_segmentation(unet, processed_image_ds.batch(BATCH_SIZE), 23)
print(f"Pixel accuracy: {pa:.4f} | mIoU: {miou:.4f}")
print(pd.Series(iou, name="IoU").round(3).to_string())

# %% [markdown]
# <a name='4-3'></a>
# ### 4.3 - Predicții

# %%
def show_predictions(dataset=None, num=1, model=None):
    model = model or unet
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image, verbose=0)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...], verbose=0))])

show_predictions(train_dataset, 6)

# %% [markdown]
# ---
# # Partea B — Sarcina 4: set de date nou (Oxford-IIIT Pet)
#
# Am ales **Oxford-IIIT Pet** (≈7 390 imagini cu pisici și câini, 37 de rase), publicat de Oxford VGG și disponibil și pe Kaggle.
# Fiecare imagine are o **trimap** pixel-cu-pixel: 1 = animal, 2 = fundal, 3 = contur.
#
# ## 4.1 Colectarea imaginilor
# Domeniu: animale de companie. Setul e variat (rase, poziții, iluminare, fundaluri).
# Pentru CPU folosim un subset (`MAX_SAMPLES`); pune `None` pentru tot setul.

# %%
import urllib.request, tarfile, shutil
from glob import glob

RAW_DIR = Path("pets_raw"); RAW_DIR.mkdir(exist_ok=True)
URLS = {
    "images": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",        # ~750 MB
    "annotations": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}
for name, url in URLS.items():
    archive = RAW_DIR / f"{name}.tar.gz"
    if not archive.exists():
        print("Descarc", url); urllib.request.urlretrieve(url, archive)
    if not (RAW_DIR / name).exists():
        print("Dezarhivez", archive)
        with tarfile.open(archive) as t:
            t.extractall(RAW_DIR, filter="data")

IMG_DIR = RAW_DIR / "images"
TRI_DIR = RAW_DIR / "annotations" / "trimaps"
names = sorted(p.stem for p in IMG_DIR.glob("*.jpg")
               if (TRI_DIR / f"{p.stem}.png").exists() and not p.name.startswith("."))
print("Perechi imagine–trimap găsite:", len(names))

# %% [markdown]
# ## 4.2 Etichetarea (măștile)
# Setul vine deja etichetat. Convertim trimap-ul în măști de clase **0..N-1**, cum cere `SparseCategoricalCrossentropy`:
#
# | trimap | clasă nouă | semnificație |
# |---|---|---|
# | 2 | 0 | fundal |
# | 1 | 1 | animal |
# | 3 | 2 | contur |
#
# *Dacă ți-ai crea propriul set:* în **CVAT** / **LabelMe** desenezi poligoane pe fiecare obiect, apoi exporți
# în format „Segmentation mask” (PNG cu o culoare/index per clasă). Pasul de conversie la indecși 0..N-1 e identic cu cel de mai jos.

# %%
CLASS_NAMES = ["fundal", "animal", "contur"]
N_CLASSES_PETS = len(CLASS_NAMES)

def trimap_to_mask(tri):
    mask = np.zeros_like(tri, dtype=np.uint8)
    mask[tri == 1] = 1
    mask[tri == 3] = 2
    return mask

ex = names[0]
ex_img = cv2.cvtColor(cv2.imread(str(IMG_DIR / f"{ex}.jpg")), cv2.COLOR_BGR2RGB)
ex_tri = cv2.imread(str(TRI_DIR / f"{ex}.png"), cv2.IMREAD_GRAYSCALE)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].imshow(ex_img); ax[0].set_title(ex)
ax[1].imshow(ex_tri); ax[1].set_title(f"trimap, valori {np.unique(ex_tri)}")
ax[2].imshow(trimap_to_mask(ex_tri), vmin=0, vmax=2); ax[2].set_title("mască 0/1/2")
for a in ax: a.axis("off")
plt.show()

# %% [markdown]
# ## 4.3 Preprocesarea cu OpenCV
# - **Redimensionare** la 128×128: imaginile cu `INTER_AREA` (anti-aliasing la micșorare), măștile cu `INTER_NEAREST` (nu se inventează clase).
# - **Curățare**: sărim peste fișierele care nu se pot citi (câteva JPEG-uri din set sunt corupte sau în alt format); `IMREAD_COLOR` aduce imaginile grayscale sau RGBA la 3 canale BGR.
# - **Normalizare** [0,1]: o facem la încărcare în `tf.data`.
# - **Augmentare** (flip orizontal): doar pe train, la antrenare.
# - **Împărțire** 80/10/10 cu seed fix.

# %%
IMG_SIZE = 128
MAX_SAMPLES = 2000          # None = tot setul (~7 390); pe CPU, 2000 e un compromis bun
rng = np.random.default_rng(42)
sel = list(rng.permutation(names))
if MAX_SAMPLES: sel = sel[:MAX_SAMPLES]

def load_and_preprocess(name):
    img = cv2.imread(str(IMG_DIR / f"{name}.jpg"), cv2.IMREAD_COLOR)
    tri = cv2.imread(str(TRI_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or tri is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tri = cv2.resize(tri, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return img, trimap_to_mask(tri)

n = len(sel); n_train = int(0.8 * n); n_val = int(0.1 * n)
splits = {"train": sel[:n_train], "val": sel[n_train:n_train + n_val], "test": sel[n_train + n_val:]}
print({k: len(v) for k, v in splits.items()})

# %% [markdown]
# ## 4.4 Salvarea și organizarea setului de date
# ```
# Dataset/
#   ├── images/{train,val,test}/
#   └── masks/{train,val,test}/
# ```
# Imaginile se salvează ca PNG (fără pierderi). Masca are același nume ca imaginea și conține valorile 0/1/2.

# %%
DATASET = Path("Dataset")
if DATASET.exists():
    shutil.rmtree(DATASET)
skipped = []
for split, items in splits.items():
    (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET / "masks" / split).mkdir(parents=True, exist_ok=True)
    for name in items:
        r = load_and_preprocess(name)
        if r is None:
            skipped.append(name); continue
        img, mask = r
        cv2.imwrite(str(DATASET / "images" / split / f"{name}.png"), img)   # BGR -> PNG corect
        cv2.imwrite(str(DATASET / "masks" / split / f"{name}.png"), mask)

for split in splits:
    ni = len(list((DATASET / "images" / split).glob("*.png")))
    nm = len(list((DATASET / "masks" / split).glob("*.png")))
    print(f"{split:5s}: {ni} imagini, {nm} măști")
print("Sărite (nu au putut fi citite):", skipped)

# %% [markdown]
# Distribuția claselor pe train (proporția pixelilor) ne arată dacă setul e dezechilibrat:

# %%
counts = np.zeros(N_CLASSES_PETS, dtype=np.int64)
for p in (DATASET / "masks" / "train").glob("*.png"):
    counts += np.bincount(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).ravel(), minlength=N_CLASSES_PETS)
print(pd.Series(counts / counts.sum(), index=CLASS_NAMES).round(3).to_string())

# %% [markdown]
# ## 4.5 Antrenarea U-Net pe setul nou
# Încărcăm datele cu `tf.data` din structura `Dataset/`, folosim aceeași funcție `unet_model` cu
# `input_size=(128,128,3)` și `n_classes=3`, plus validare, EarlyStopping și salvarea celui mai bun model.

# %%
AUTOTUNE = tf.data.AUTOTUNE
BATCH_PETS = 16

def make_ds(split, training=False):
    imgs = sorted(glob(str(DATASET / "images" / split / "*.png")))
    msks = [p.replace(f"images{os.sep}{split}", f"masks{os.sep}{split}") for p in imgs]
    ds = tf.data.Dataset.from_tensor_slices((imgs, msks))

    def _load(ip, mp):
        img = tf.cast(tf.image.decode_png(tf.io.read_file(ip), channels=3), tf.float32) / 255.0
        m = tf.image.decode_png(tf.io.read_file(mp), channels=1)
        img.set_shape([IMG_SIZE, IMG_SIZE, 3]); m.set_shape([IMG_SIZE, IMG_SIZE, 1])
        return img, m

    def _augment(img, m):
        flip = tf.random.uniform(()) > 0.5
        img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
        m = tf.cond(flip, lambda: tf.image.flip_left_right(m), lambda: m)
        return img, m

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(1000).map(_augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_PETS).prefetch(AUTOTUNE)

train_pets, val_pets, test_pets = make_ds("train", True), make_ds("val"), make_ds("test")

for x, y in train_pets.take(1):
    print(x.shape, x.dtype, float(tf.reduce_max(x)), "|", y.shape, y.dtype, np.unique(y.numpy()))
    display([x[0], y[0]])

# %%
unet_pets = unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3), n_filters=32, n_classes=N_CLASSES_PETS)
unet_pets.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

EPOCHS_PETS = 3 if QUICK_RUN else 25
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("models/pets_unet_best.keras", monitor="val_loss", save_best_only=True),
]
hist_pets = unet_pets.fit(train_pets, validation_data=val_pets, epochs=EPOCHS_PETS, callbacks=callbacks)

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k in ["accuracy", "val_accuracy"]: ax[0].plot(hist_pets.history[k], label=k)
for k in ["loss", "val_loss"]:         ax[1].plot(hist_pets.history[k], label=k)
ax[0].set_title("Accuracy"); ax[1].set_title("Loss")
for a in ax: a.legend(); a.set_xlabel("epoca")
plt.show()

# %% [markdown]
# ### Evaluarea pe setul de test (date nevăzute la antrenare)

# %%
pa, iou, miou = evaluate_segmentation(unet_pets, test_pets, N_CLASSES_PETS)
print(f"TEST — Pixel accuracy: {pa:.4f} | mIoU: {miou:.4f}")
print(pd.Series(iou, index=CLASS_NAMES, name="IoU").round(4).to_string())

# %%
show_predictions(test_pets, 5, model=unet_pets)

# %% [markdown]
# ## Concluzii
# - Am construit U-Net de la zero (encoder, decoder, skip connections) și am trecut testele din `test_utils`.
# - **CARLA:** 23 de clase. Am corectat două probleme din original: perechile imagine–mască nesortate și dubla normalizare /255.
# - **Oxford-IIIT Pet:** am parcurs tot fluxul cerut: colectare → măști (trimap → 0/1/2) → preprocesare OpenCV →
#   împărțire 80/10/10 → structura `Dataset/` → antrenare cu validare → evaluare pe test cu mIoU.
# - Clasa „contur” e cea mai grea (puțini pixeli, subțire). Ce s-ar putea îmbunătăți: rezoluție mai mare, mai multe date sau augmentări,
#   Dice loss sau ponderi pe clase, un encoder preantrenat.
