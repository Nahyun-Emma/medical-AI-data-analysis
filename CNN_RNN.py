#@@ 이미지 형태 변환하기

from elice_utils import EliceUtils

elice_utils = EliceUtils()

from PIL import Image


def crop(img, coordinates):
    # TODO: [지시사항 1번] 이미지를 자르는 코드를 완성하세요.
    img_crop = img.crop(coordinates)

    return img_crop


def rotate(img, angle, expand=False):
    # TODO: [지시사항 2번] 이미지를 회전하는 코드를 완성하세요.
    img_rotate = img.rotate(angle, expand=expand)

    return img_rotate


def resize(img, new_size):
    # TODO: [지시사항 3번] 이미지 크기를 변경하는 코드를 완성하세요.
    img_resize = img.resize(new_size)

    return img_resize


def shearing(img, shear_factor):
    # TODO: [지시사항 4번] 이미지를 전단 변환하는 코드를 완성하세요.
    img_shearing = img.transform((int(img.size[0] * (1 + shear_factor)), img.size[1]),
                                 Image.AFFINE, (1, -shear_factor, 0, 0, 1, 0))  # 이코드는 잘 기억해놓기

    return img_shearing


def show_image(img, name):
    img.save(name)
    elice_utils.send_image(name)


def main():
    img = Image.open("Lenna.png")

    # TODO: [지시사항 5번] 지시사항에 따라 적절한 이미지 변환을 수행하세요.

    # 이미지 자르기
    img_crop = crop(img, (150, 200, 450, 300))

    # 이미지 회전하기
    img_rotate = rotate(img, 160, expand=True)

    # 이미지 크기 바꾸기
    img_resize = resize(img, (640, 360))

    # 이미지 전단 변환
    img_shearing = shearing(img, 0.8)

    print("=" * 50, "Crop 결과", "=" * 50)
    show_image(img_crop, "crop.png")

    print("=" * 50, "Rotate 결과", "=" * 50)
    show_image(img_rotate, "rotate.png")

    print("=" * 50, "Resize 결과", "=" * 50)
    show_image(img_resize, "resize.png")

    print("=" * 50, "Shearing 결과", "=" * 50)
    show_image(img_shearing, "shearing.png")

    return img_crop, img_rotate, img_resize, img_shearing


if __name__ == "__main__":
    main()


#@@ 이미지 색상 변환하기
from elice_utils import EliceUtils

elice_utils = EliceUtils()

from PIL import Image
from PIL import ImageEnhance


def change_brightness(img, factor):
    # TODO: [지시사항 1번] 이미지의 밝기를 변화시키는 코드를 완성하세요.
    bright_enhancer = ImageEnhance.Brightness(img)
    img_bright = bright_enhancer.enhance(factor)

    return img_bright


def change_contrast(img, factor):
    # TODO: [지시사항 2번] 이미지의 대조를 변화시키는 코드를 완성하세요.
    contrast_enhancer = ImageEnhance.Contrast(img)
    img_contrast = contrast_enhancer.enhance(factor)

    return img_contrast


def change_grayscale(img):
    # TODO: [지시사항 3번] 이미지를 흑백 이미지로 변경하는 코드를 완성하세요.
    img_gray = img.convert("L")

    return img_gray


def show_image(img, name):
    img.save(name)
    elice_utils.send_image(name)


def main():
    img = Image.open("Lenna.png")

    # TODO: [지시사항 4번] 지시사항에 따라 적절한 이미지 변환을 수행하세요.

    # 이미지 밝게 하기
    img_bright = change_brightness(img, 1.5)

    # 이미지 어둡게 하기
    img_dark = change_brightness(img, 0.2)

    # 이미지 대조 늘리기
    img_high_contrast = change_contrast(img, 3)

    # 이미지 대조 줄이기
    img_low_contrast = change_contrast(img, 0.1)

    # 이미지 흑백 변환
    img_gray = change_grayscale(img)

    print("=" * 50, "밝은 이미지", "=" * 50)
    show_image(img_bright, "bright.png")

    print("=" * 50, "어두운 이미지", "=" * 50)
    show_image(img_dark, "dark.png")

    print("=" * 50, "강한 대조 이미지", "=" * 50)
    show_image(img_high_contrast, "high_contrast.png")

    print("=" * 50, "약한 대조 이미지", "=" * 50)
    show_image(img_low_contrast, "low_contrast.png")

    print("=" * 50, "흑백 이미지", "=" * 50)
    show_image(img_gray, "gray.png")

    return img_bright, img_dark, img_high_contrast, img_low_contrast, img_gray


if __name__ == "__main__":
    main()

#@@ 이미지 필터 변환하기
from elice_utils import EliceUtils

elice_utils = EliceUtils()

from PIL import Image
from PIL import ImageFilter


def sharpening(img):
    # TODO: [지시사항 1번] 이미지에 샤프닝 필터를 적용시키는 코드를 완성하세요.
    img_sharpen = img.filter(ImageFilter.SHARPEN)

    return img_sharpen


def blur(img):
    # TODO: [지시사항 2번] 이미지에 블러 필터를 적용시키는 코드를 완성하세요.
    img_blur = img.filter(ImageFilter.BLUR)

    return img_blur


def detect_edge(img):
    # TODO: [지시사항 3번] 이미지의 경계선을 탐지하는 코드를 완성하세요.
    img_edge = img.filter(ImageFilter.FIND_EDGES)

    return img_edge


def show_image(img, name):
    img.save(name)
    elice_utils.send_image(name)


def main():
    img = Image.open("Lenna.png")

    # TODO: [지시사항 4번] 지시사항에 따라 적절한 이미지 변환을 수행하세요.

    # 이미지 샤프닝 한번 적용하기
    img_sharpen_1 = sharpening(img)

    # 이미지 샤프닝 5번 적용하기
    img_sharpen_5 = sharpening(img)
    img_sharpen_5 = sharpening(img_sharpen_5)
    img_sharpen_5 = sharpening(img_sharpen_5)
    img_sharpen_5 = sharpening(img_sharpen_5)
    img_sharpen_5 = sharpening(img_sharpen_5)

    # 이미지 블러 한번 적용하기
    img_blur_1 = blur(img)

    # 이미지 블러 5번 적용하기
    img_blur_5 = blur(img)
    img_blur_5 = blur(img_blur_5)
    img_blur_5 = blur(img_blur_5)
    img_blur_5 = blur(img_blur_5)
    img_blur_5 = blur(img_blur_5)

    # 이미지 경계선 찾기
    img_edge = detect_edge(img)

    print("=" * 50, "샤프닝 한번 적용한 이미지", "=" * 50)
    show_image(img_sharpen_1, "sharpen_1.png")

    print("=" * 50, "샤프닝 다섯번 적용한 이미지", "=" * 50)
    show_image(img_sharpen_5, "sharpen_5.png")

    print("=" * 50, "블러 한번 적용한 이미지", "=" * 50)
    show_image(img_blur_1, "blur_1.png")

    print("=" * 50, "블러 다섯번 적용한 이미지", "=" * 50)
    show_image(img_blur_5, "blur_5.png")

    print("=" * 50, "경계선 이미지", "=" * 50)
    show_image(img_edge, "edge.png")

    return img_sharpen_1, img_sharpen_5, img_blur_1, img_blur_5, img_edge


if __name__ == "__main__":
    main()

#@@ 딥러닝 모델 학습을 위한 이미지 불러오기
from elice_utils import EliceUtils

elice_utils = EliceUtils()

from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np


def show_plot(img, title=" "):
    plt.title(title)
    plt.imshow(img)
    plt.savefig("tmp.png")
    elice_utils.send_image("tmp.png")


def load_image(path, name):
    # TODO: [지시사항 1번] 이미지를 불러오는 함수를 완성하세요
    path = "dataset/val/dogs"
    name = "dog.0.jpg"
    img = Image.open(os.path.join(path, name))
    return img


def main():
    data_path = "dataset/val/dogs"

    # 이미지를 불러와 plt를 이용하여 출력합니다
    names = os.listdir(data_path)
    img = load_image(data_path, names[0])

    # 원본 이미지를 출력
    show_plot(img, "PIL original image")

    # TODO: [지시사항 2번] 지시사항에 따라 이미지의 크기를 확인하는 코드를 완성하세요.
    # PIL을 통해 이미지 크기 확인
    pil_size = img.size
    print("PIL을 통한 이미지 크기:", pil_size)

    # PIL 이미지를 numpy 배열로 변환
    np_img = np.array(img)

    # numpy 배열의 shape 확인
    np_shape = np_img.shape
    print("numpy 배열 shape:", np_shape)
    show_plot(np_img, "Numpy array image")

    # TODO: [지시사항 3번] PIL과 numpy를 이용하여 이미지를 다루는 코드를 완성하세요.
    # PIL.Image에서 x=10, y=20 의 픽셀값 가져오기
    pil_pix = img.load()[10, 20]

    # numpy 배열에서 x=10, y=20 의 픽셀값을 가져오세요
    np_pix = np_img[20, 10]
    print("PIL의 픽셀값: {}, numpy의 픽셀값: {}".format(pil_pix, np_pix))

    # PIL을 이용하여 이미지의 크기를 (224,224)로 변형하세요.
    resized_img = img.resize((224, 224))

    # resize된 이미지 출력
    show_plot(resized_img, "Resized image")
    print("resize 결과 사이즈:", resized_img.size)

    return pil_size, np_img, np_shape, np_pix, resized_img


if __name__ == "__main__":
    main()


#@@ 커스텀 데이터셋 불러오기
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_path = "dataset"

batch_size = 2
img_height = 180
img_width = 180


# path의 데이터를 ImageDataGenerator로 불러와주는 함수
def get_dataset(path, datagen):
    data_set = datagen.flow_from_directory(path,
                                           target_size=(img_width, img_height),
                                           batch_size=batch_size,
                                           class_mode='categorical')
    return data_set


def main():
    # TODO: [지시사항 1번] 정규화 과정이 없는 ImageDataGenerator를 만드세요.
    first_gen = ImageDataGenerator()
    first_set = get_dataset(os.path.join(data_path, "val"), first_gen)
    x, y = first_set.__next__()

    print("\n1. 데이터 제너레이터 만들기")
    print("first_set")
    print("x: {}, y: {}".format(x.shape, y.shape))
    print(x[0][0][0])  # 픽셀이 0~255의 값을 가짐

    # TODO: [지시사항 2번] 픽셀값을 0~1의 값으로 정규화 하는 ImageDataGenerator를 만드세요.
    second_gen = ImageDataGenerator(rescale=1 / 255)
    second_set = get_dataset(os.path.join(data_path, "val"), second_gen)
    x, y = second_set.__next__()

    print("\n2. 데이터 제너레이터에 정규화 추가하기")
    print("second_set")
    print("x: {}, y: {}".format(x.shape, y.shape))
    print(x[0][0][0])  # 픽셀이 0~1의 값을 가지는 것을 확인하세요

    # TODO: [지시사항 3번] 실제 학습을 위한 ImageDataGenerator를 만드세요.
    # 학습 데이터를 위한 ImageDataGenerator를 만드세요.
    train_gen = ImageDataGenerator(rescale=1 / 255)

    # 학습 데이터셋을 불러오도록 경로명을 설정하세요.
    train_set = get_dataset(os.path.join(data_path, "train"), train_gen)

    # 검증 데이터를 위한 ImageDataGenerator를 만드세요.
    valid_gen = ImageDataGenerator(rescale=1 / 255)

    # 검증 데이터셋을 불러오도록 경로명을 설정하세요.
    valid_set = get_dataset(os.path.join(data_path, "val"), train_gen)

    print("\n3. 실제 학습을 위한 데이터 제너레이터 작성")
    print("학습 데이터의 길이: ", len(train_set))
    print("검증 데이터의 길이: ", len(valid_set))

    return first_gen, second_gen, train_gen, train_set, valid_gen, valid_set


if __name__ == "__main__":
    main()

#@@ Padding, Stride와 Layer size
import tensorflow as tf
from tensorflow.keras import layers, Sequential


# TODO: [지시사항 1번] 지시사항 대로 Conv2D 하나로 이루어진 모델을 완성하세요
def build_model1(input_shape):
    model = layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding="same",
                          activation="relu",
                          input_shape=input_shape[1:])

    return model


# TODO: [지시사항 2번] 지시사항 대로 Conv2D 두개로 이루어진 모델을 완성하세요
def build_model2(input_shape):
    model = Sequential(
        [layers.Conv2D(4, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=input_shape[1:]),
         layers.Conv2D(4, kernel_size=(3, 3), strides=(1, 1), padding='same')])

    return model


# TODO: [지시사항 3번] 지시사항 대로 Conv2D 세개로 이루어진 모델을 완성하세요
def build_model3(input_shape):
    model = Sequential()

    model.add(layers.Conv2D(2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            input_shape=input_shape[1:]))  # Sequential의 기능 add
    model.add(layers.Conv2D(4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(layers.Conv2D(8, kernel_size=(3, 3), strides=(1, 1)))

    return model


def main():
    input_shape = (1, 5, 5, 1)

    model1 = build_model1(input_shape)
    model2 = build_model2(input_shape)
    model3 = build_model3(input_shape)

    x = tf.ones(input_shape)
    print("model1을 통과한 결과:", model1(x).shape)
    print("model2을 통과한 결과:", model2(x).shape)
    print("model3을 통과한 결과:", model3(x).shape)


if __name__ == "__main__":
    main()


#@@ MLP로 이미지 데이터 학습하기
from elice_utils import EliceUtils

elice_utils = EliceUtils()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

SEED = 2021


def load_cifar10_dataset():
    train_X = np.load("./dataset/cifar10_train_X.npy")
    train_y = np.load("./dataset/cifar10_train_y.npy")
    test_X = np.load("./dataset/cifar10_test_X.npy")
    test_y = np.load("./dataset/cifar10_test_y.npy")

    train_X, test_X = train_X / 255.0, test_X / 255.0  # image pixcel의 크기가 0~255임. 그래서 /255하면 정규화해주는 과정임!

    return train_X, train_y, test_X, test_y


def build_mlp_model(img_shape, num_classes=10):
    model = Sequential()

    model.add(Input(shape=img_shape))  # Conv layer의 input layer개념

    # TODO: [지시사항 1번] 모델을 완성하세요.
    model.add(layers.Flatten())  # 2차원 이미지를 1차원으로 변형
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def plot_history(hist):
    train_loss = hist.history["loss"]
    train_acc = hist.history["accuracy"]
    valid_loss = hist.history["val_loss"]
    valid_acc = hist.history["val_accuracy"]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train', 'Valid'], loc='upper right')
    plt.savefig("loss.png")
    elice_utils.send_image("loss.png")

    fig = plt.figure(figsize=(8, 6))
    plt.plot(train_acc)
    plt.plot(valid_acc)
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig("accuracy.png")
    elice_utils.send_image("accuracy.png")


def main(model=None, epochs=10):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    train_X, train_y, test_X, test_y = load_cifar10_dataset()
    img_shape = train_X[0].shape

    # TODO: [지시사항 2번] Adam optimizer를 설정하세요.
    optimizer = Adam(learning_rate=0.001)

    mlp_model = model
    if model is None:
        mlp_model = build_mlp_model(img_shape)

    # TODO: [지시사항 3번] 모델의 optimizer, 손실 함수, 평가 지표를 설정하세요.
    mlp_model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    # TODO: [지시사항 4번] 모델 학습을 위한 hyperparameter를 설정하세요.
    hist = mlp_model.fit(train_X, train_y, epochs=epochs, batch_size=64, validation_split=0.2, shuffle=True, verbose=2)
    # validation_split : train dataset내에서 별도의 validation dataset을 만드는 것 (테스트셋과는 다름!!!)
    # shuffle=True : 모델이 학습할 때 순서까지 외워버리는 것을 방지하기 위함.

    plot_history(hist)
    test_loss, test_acc = mlp_model.evaluate(test_X, test_y)
    print("Test Loss: {:.5f}, Test Accuracy: {:.3f}%".format(test_loss, test_acc * 100))

    return optimizer, hist


if __name__ == "__main__":
    main()

#@@ MLP와 CNN 모델 비교
from elice_utils import EliceUtils

elice_utils = EliceUtils()

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

SEED = 2021


def load_cifar10_dataset():
    train_X = np.load("./dataset/cifar10_train_X.npy")
    train_y = np.load("./dataset/cifar10_train_y.npy")
    test_X = np.load("./dataset/cifar10_test_X.npy")
    test_y = np.load("./dataset/cifar10_test_y.npy")

    train_X, test_X = train_X / 255.0, test_X / 255.0

    return train_X, train_y, test_X, test_y


def build_mlp_model(img_shape, num_classes=10):
    model = Sequential()

    model.add(Input(shape=img_shape))

    # TODO: [지시사항 1번] MLP 모델을 완성하세요.
    model.add(layers.Flatten())  # 2차원 이미지를 1차원으로 변형
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def build_cnn_model(img_shape, num_classes=10):
    model = Sequential()

    # TODO: [지시사항 2번] CNN 모델을 완성하세요.
    model.add(layers.Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=
    (img_shape), activation='relu'))  # stride 기본값은 (1,1)
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(layers.MaxPool2D(2))
    # model.add(layers.MaxPool2D((2,2)))
    # pooling layer는 MaxPool2D를 통해 구현/ pool_size, strides를 파라미터로 가지는데, 2배로 줄이려면 두 파라미터 모두 (2,2), (2,2)로 설정하면 됨. 근데, strides 설정하지 않으면, pool_size를 따라감, (n,n)식으로 가로세로가 똑같은 경우는 pool_size=n, strides=n 이런식으로 작성 가능
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', strides=(2, 2),
                            activation='relu'))  # strides=(2,2) == maxpooling 2배랑 같은 효과
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu'))
    model.add(layers.MaxPool2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def plot_history(hist):
    train_loss = hist.history["loss"]
    train_acc = hist.history["accuracy"]
    valid_loss = hist.history["val_loss"]
    valid_acc = hist.history["val_accuracy"]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train', 'Valid'], loc='upper right')
    plt.savefig("loss.png")
    elice_utils.send_image("loss.png")

    fig = plt.figure(figsize=(8, 6))
    plt.plot(train_acc)
    plt.plot(valid_acc)
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig("accuracy.png")
    elice_utils.send_image("accuracy.png")


def run_model(model, train_X, train_y, test_X, test_y, epochs=10):
    # TODO: [지시사항 3번] Adam optimizer를 설정하세요.
    optimizer = Adam(learning_rate=0.001)

    model.summary()
    # TODO: [지시사항 4번] 모델의 optimizer, 손실 함수, 평가 지표를 설정하세요.
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TODO: [지시사항 5번] 모델 학습을 위한 hyperparameter를 설정하세요.
    hist = model.fit(train_X, train_y, epochs=epochs, batch_size=64, validation_split=0.2, shuffle=True, verbose=2)

    plot_history(hist)
    test_loss, test_acc = model.evaluate(test_X, test_y)
    print("Test Loss: {:.5f}, Test Accuracy: {:.3f}%".format(test_loss, test_acc * 100))

    return optimizer, hist


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    train_X, train_y, test_X, test_y = load_cifar10_dataset()
    img_shape = train_X[0].shape

    mlp_model = build_mlp_model(img_shape)
    cnn_model = build_cnn_model(img_shape)

    print("=" * 30, "MLP 모델", "=" * 30)
    run_model(mlp_model, train_X, train_y, test_X, test_y)

    print()
    print("=" * 30, "CNN 모델", "=" * 30)
    run_model(cnn_model, train_X, train_y, test_X, test_y)


if __name__ == "__main__":
    main()

#@@ VGG16 구현
import tensorflow as tf
from tensorflow.keras import Sequential, layers


def build_vgg16():
    # Sequential 모델 선언
    model = Sequential()

    # TODO: [지시시항 1번] 첫번째 Block을 완성하세요.
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2))

    # TODO: [지시시항 2번] 두번째 Block을 완성하세요.
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2))

    # TODO: [지시시항 3번] 세번째 Block을 완성하세요.
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2))

    # TODO: [지시시항 4번] 네번째 Block을 완성하세요.
    model.add(layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2))

    # TODO: [지시시항 5번] 다섯번째 Block을 완성하세요.
    model.add(layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2))

    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dense(1000, activation="softmax"))

    return model


def main():
    model = build_vgg16()
    model.summary()


if __name__ == "__main__":
    main()

#@@ ResNet 구현하기
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class ResidualBlock(Model):
    def __init__(self, num_kernels, kernel_size):
        super(ResidualBlock, self).__init__()

        # TODO: [지시사항 1번] 2개의 Conv2D Layer를 지시사항에 따라 추가하세요.
        self.conv1 = layers.Conv2D(num_kernels, kernel_size=kernel_size, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(num_kernels, kernel_size=kernel_size, padding='same', activation='relu')

        self.relu = layers.Activation("relu")

        # TODO: [지시사항 1번] Add Layer를 추가하세요.
        self.add = layers.Add()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)

        x = self.add([x, input_tensor])  # 두 값을 더하는 과정 : 이부분이 중요함!!
        x = self.relu(x)

        return x


def build_resnet(input_shape, num_classes):
    model = Sequential()

    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.MaxPool2D(2))

    model.add(ResidualBlock(64, (3, 3)))
    model.add(ResidualBlock(64, (3, 3)))
    model.add(ResidualBlock(64, (3, 3)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def main():
    input_shape = (32, 32, 3)
    num_classes = 10

    model = build_resnet(input_shape, num_classes)
    model.summary()


if __name__ == "__main__":
    main()


#@@ vanila RNN 모델 만들기
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential


# TODO: [지시사항 1번] 첫번째 모델을 완성하세요.
def build_model1():
    model = Sequential()

    model.add(layers.Embedding(10, 5))  # input_dim(전체단어 갯수), output_dim(벡터길이)
    model.add(layers.SimpleRNN(3))

    return model


# TODO: [지시사항 2번] 두번째 모델을 완성하세요.
def build_model2():
    model = Sequential()

    model.add(layers.Embedding(256, 100))
    model.add(layers.SimpleRNN(20))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def main():
    model1 = build_model1()
    print("=" * 20, "첫번째 모델", "=" * 20)
    model1.summary()

    print()

    model2 = build_model2()
    print("=" * 20, "두번째 모델", "=" * 20)
    model2.summary()


if __name__ == "__main__":
    main()


#@@ Vanilla RNN으로 IMDb 데이터 학습하기
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import imdb  # imdb 데이터셋 불러오기 위한 모듈
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data(num_words, max_len):
    # imdb 데이터셋을 불러옵니다. 데이터셋에서 단어는 num_words 개를 가져옵니다.
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

    # 단어 개수가 다른 문장들을 Padding을 추가하여
    # 단어가 가장 많은 문장의 단어 개수로 통일합니다.
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    return X_train, X_test, y_train, y_test


def build_rnn_model(num_words, embedding_len):
    model = Sequential()

    # TODO: [지시사항 1번] 지시사항에 따라 모델을 완성하세요.
    model.add(layers.Embedding(num_words, embedding_len))
    model.add(layers.SimpleRNN(16))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def main(model=None, epochs=5):
    # IMDb 데이터셋에서 가져올 단어의 개수
    num_words = 6000

    # 각 문장이 가질 수 있는 최대 단어 개수
    max_len = 130

    # 임베딩 된 벡터의 길이
    embedding_len = 100

    # IMDb 데이터셋을 불러옵니다.
    X_train, X_test, y_train, y_test = load_data(num_words, max_len)

    if model is None:
        model = build_rnn_model(num_words, embedding_len)

    # TODO: [지시사항 2번] 모델 학습을 위한 optimizer와 loss 함수를 설정하세요.
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # TODO: [지시사항 3번] 모델 학습을 위한 hyperparameter를 설정하세요.
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=100, validation_split=0.2, shuffle=True, verbose=2)

    # 모델을 테스트 데이터셋으로 테스트합니다.
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print()
    print("테스트 Loss: {:.5f}, 테스트 정확도: {:.3f}%".format(test_loss, test_acc * 100))

    return optimizer, hist


if __name__ == "__main__":
    main()

#@@ Vanilla RNN을 통한 항공 승객 수 분석

from elice_utils import EliceUtils

elice_utils = EliceUtils()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(window_size):
    raw_data = pd.read_csv("./airline-passengers.csv")
    raw_passengers = raw_data["Passengers"].to_numpy()

    # 데이터의 평균과 표준편차 값으로 정규화(표준화) 합니다.
    mean_passenger = raw_passengers.mean()
    stdv_passenger = raw_passengers.std(ddof=0)
    raw_passengers = (raw_passengers - mean_passenger) / stdv_passenger
    data_stat = {"month": raw_data["Month"], "mean": mean_passenger, "stdv": stdv_passenger}

    # window_size 개의 데이터를 불러와 입력 데이터(X)로 설정하고
    # window_size보다 한 시점 뒤의 데이터를 예측할 대상(y)으로 설정하여
    # 데이터셋을 구성합니다.
    X, y = [], []
    for i in range(len(raw_passengers) - window_size):
        cur_passenger = raw_passengers[i:i + window_size]
        target = raw_passengers[i + window_size]

        X.append(list(cur_passenger))
        y.append(target)

    # X와 y를 numpy array로 변환합니다.
    X = np.array(X)
    y = np.array(y)

    # 각 입력 데이터는 sequence 길이가 window_size이고, featuer 개수는 1개가 되도록
    # 마지막에 새로운 차원을 추가합니다.
    # 즉, (전체 데이터 개수, window_size) -> (전체 데이터 개수, window_size, 1)이 되도록 변환합니다.
    X = X[:, :, np.newaxis]

    # 학습 데이터는 전체 데이터의 80%, 테스트 데이터는 20%로 설정합니다.
    total_len = len(X)
    train_len = int(total_len * 0.8)

    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    return X_train, X_test, y_train, y_test, data_stat


def build_rnn_model(window_size):
    model = Sequential()

    # TODO: [지시사항 1번] SimpleRNN 기반 모델을 구성하세요.
    model.add(layers.SimpleRNN(4, input_shape=(window_size, 1)))  # 이 코드 중요함
    model.add(layers.Dense(1))
    return model


def plot_result(X_true, y_true, y_pred, data_stat):
    # 표준화된 결과를 다시 원래 값으로 변환합니다.
    y_true_orig = (y_true * data_stat["stdv"]) + data_stat["mean"]
    y_pred_orig = (y_pred * data_stat["stdv"]) + data_stat["mean"]

    # 테스트 데이터에서 사용한 날짜들만 가져옵니다.
    test_month = data_stat["month"][-len(y_true):]

    # 모델의 예측값을 실제값과 함께 그래프로 그립니다.
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.plot(y_true_orig, color="b", label="True")
    ax.plot(y_pred_orig, color="r", label="Prediction")
    ax.set_xticks(list(range(len(test_month))))
    ax.set_xticklabels(test_month, rotation=45)
    ax.set_title("RNN Result")
    ax.legend(loc="upper left")
    plt.savefig("airline_rnn.png")
    elice_utils.send_image("airline_rnn.png")


def main(model=None, epochs=10):
    tf.random.set_seed(2022)

    window_size = 4
    X_train, X_test, y_train, y_test, data_stat = load_data(window_size)

    if model is None:
        model = build_rnn_model(window_size)

    # TODO: [지시사항 2번] 모델 학습을 위한 optimizer와 loss 함수를 설정하세요.
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    # TODO: [지시사항 3번] 모델 학습을 위한 hyperparameter를 설정하세요.
    hist = model.fit(X_train, y_train, batch_size=8, epochs=epochs, shuffle=True, verbose=2)

    # 테스트 데이터셋으로 모델을 테스트합니다.
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print()
    print("테스트 MSE: {:.5f}".format(test_loss))
    print()

    # 모델의 예측값과 실제값을 그래프로 그립니다.
    y_pred = model.predict(X_test)
    plot_result(X_test, y_test, y_pred, data_stat)

    return optimizer, hist


if __name__ == "__main__":
    main()

#@@ 심층 Vanilla RNN 모델
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np


def load_data(num_data, window_size):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, num_data, 1)

    time = np.linspace(0, 1, window_size + 1)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.1 * np.sin((time - offsets2) * (freq2 * 10 + 10))
    series += 0.1 * (np.random.rand(num_data, window_size + 1) - 0.5)

    num_train = int(num_data * 0.8)
    X_train, y_train = series[:num_train, :window_size], series[:num_train, -1]
    X_test, y_test = series[num_train:, :window_size], series[num_train:, -1]

    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]

    return X_train, X_test, y_train, y_test


def build_rnn_model(window_size):
    model = Sequential()

    # TODO: [지시사항 1번] SimpleRNN 기반 모델을 구성하세요.
    model.add(layers.SimpleRNN(20, input_shape=(window_size, 1)))
    model.add(layers.Dense(1))
    return model


def build_deep_rnn_model(window_size):
    model = Sequential()

    # TODO: [지시사항 2번] 여러개의 SimpleRNN을 가지는 모델을 구성하세요.
    model.add(layers.SimpleRNN(20, return_sequences=True, input_shape=(window_size, 1)))
    model.add(layers.SimpleRNN(20))  # 위 코드가 return sequnces=True로 설정됨으로써, 각 시점의 출력값을 다 활용할 수 있게 됨.
    model.add(layers.Dense(1))

    return model


def run_model(model, X_train, X_test, y_train, y_test, epochs=20, name=None):
    # TODO: [지시사항 3번] 모델 학습을 위한 optimizer와 loss 함수를 설정하세요.
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    # TODO: [지시사항 4번] 모델 학습을 위한 hyperparameter를 설정하세요.
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=256, shuffle=True, verbose=2)

    # 테스트 데이터셋으로 모델을 테스트합니다.
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print("[{}] 테스트 MSE: {:.5f}".format(name, test_loss))
    print()

    return optimizer, hist


def main():
    tf.random.set_seed(2022)
    np.random.seed(2022)

    window_size = 50
    X_train, X_test, y_train, y_test = load_data(10000, window_size)

    rnn_model = build_rnn_model(window_size)
    run_model(rnn_model, X_train, X_test, y_train, y_test, name="RNN")

    deep_rnn_model = build_deep_rnn_model(window_size)
    run_model(deep_rnn_model, X_train, X_test, y_train, y_test, name="Deep RNN")


if __name__ == "__main__":
    main()

#@@ Encoder-Decoder 구조
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Sequential, Input


class EncoderDecoder(Model):
    def __init__(self, hidden_dim, encoder_input_shape, decoder_input_shape, num_classes):
        super(EncoderDecoder, self).__init__()

        # TODO: [지시사항 1번] SimpleRNN으로 이루어진 Encoder를 정의하세요.
        self.encoder = layers.SimpleRNN(hidden_dim, return_state=True, input_shape=encoder_input_shape)

        # TODO: [지시사항 2번] SimpleRNN으로 이루어진 Decoder를 정의하세요.
        self.decoder = layers.SimpleRNN(hidden_dim, return_sequences=True, input_shape=decoder_input_shape)

        self.dense = layers.Dense(num_classes, activation="softmax")

    def call(self, encoder_inputs, decoder_inputs):
        # TODO: [지시사항 3번] Encoder에 입력값을 넣어 Decoder의 초기 state로 사용할 state를 얻어내세요.
        _, encoder_state = self.encoder(encoder_inputs)

        # TODO: [지시사항 4번] Decoder에 입력값을 넣고, 초기 state는 Encoder에서 얻어낸 state로 설정하세요.
        decoder_outputs = self.decoder(decoder_inputs, initial_state=[encoder_state])

        outputs = self.dense(decoder_outputs)

        return outputs


def main():
    # hidden state의 크기
    hidden_dim = 20

    # Encoder에 들어갈 각 데이터의 모양
    encoder_input_shape = (10, 1)

    # Decoder에 들어갈 각 데이터의 모양
    decoder_input_shape = (30, 1)

    # 분류한 클래스 개수
    num_classes = 5

    # Encoder-Decoder 모델을 만듭니다.
    model = EncoderDecoder(hidden_dim, encoder_input_shape, decoder_input_shape, num_classes)

    # 모델에 넣어줄 가상의 데이터를 생성합니다.
    encoder_x, decoder_x = tf.random.uniform(shape=encoder_input_shape), tf.random.uniform(shape=decoder_input_shape)
    encoder_x, decoder_x = tf.expand_dims(encoder_x, axis=0), tf.expand_dims(decoder_x, axis=0)
    y = model(encoder_x, decoder_x)

    # 모델의 정보를 출력합니다.
    model.summary()


if __name__ == "__main__":
    main()

#@@ RNN의 장기 의존성 문제 확인
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np


def load_data(window_size):
    raw_data = pd.read_csv("./daily-min-temperatures.csv")
    raw_temps = raw_data["Temp"]

    mean_temp = raw_temps.mean()
    stdv_temp = raw_temps.std(ddof=0)
    raw_temps = (raw_temps - mean_temp) / stdv_temp

    X, y = [], []
    for i in range(len(raw_temps) - window_size):
        cur_temps = raw_temps[i:i + window_size]
        target = raw_temps[i + window_size]

        X.append(list(cur_temps))
        y.append(target)

    X = np.array(X)
    y = np.array(y)
    X = X[:, :, np.newaxis]

    total_len = len(X)
    train_len = int(total_len * 0.8)

    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    return X_train, X_test, y_train, y_test


def build_rnn_model(window_size):
    model = Sequential()

    # TODO: [지시사항 1번] Simple RNN과 Fully-connected Layer로 구성된 모델을 완성하세요.
    model.add(layers.SimpleRNN(128, input_shape=(window_size, 1)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))

    return model


def build_lstm_model(window_size):
    model = Sequential()

    # TODO: [지시사항 2번] LSTM과 Fully-connected Layer로 구성된 모델을 완성하세요.
    model.add(layers.LSTM(128, input_shape=(window_size, 1)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))

    return model


def build_gru_model(window_size):
    model = Sequential()

    # TODO: [지시사항 3번] GRU와 Fully-connected Layer로 구성된 모델을 완성하세요.
    model.add(layers.GRU(128, input_shape=(window_size, 1)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    return model


def run_model(model, X_train, X_test, y_train, y_test, epochs=10, model_name=None):
    # TODO: [지시사항 4번] 모델 학습을 위한 optimizer와 loss 함수를 설정하세요.
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    # TODO: [지시사항 5번] 모델 학습을 위한 hyperparameter를 설정하세요.
    hist = model.fit(X_train, y_train, batch_size=64, epochs=epochs, shuffle=True, verbose=2)

    # 테스트 데이터셋으로 모델을 테스트합니다.
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    return test_loss, optimizer, hist


def main(window_size):
    tf.random.set_seed(2022)
    X_train, X_test, y_train, y_test = load_data(window_size)

    rnn_model = build_rnn_model(window_size)
    lstm_model = build_lstm_model(window_size)
    gru_model = build_gru_model(window_size)

    rnn_test_loss, _, _ = run_model(rnn_model, X_train, X_test, y_train, y_test, model_name="RNN")
    lstm_test_loss, _, _ = run_model(lstm_model, X_train, X_test, y_train, y_test, model_name="LSTM")
    gru_test_loss, _, _ = run_model(gru_model, X_train, X_test, y_train, y_test, model_name="GRU")

    return rnn_test_loss, lstm_test_loss, gru_test_loss


if __name__ == "__main__":
    # 10일치 데이터를 보고 다음날의 기온을 예측합니다.
    rnn_10_test_loss, lstm_10_test_loss, gru_10_test_loss = main(10)

    # 300일치 데이터를 보고 다음날의 기온을 예측합니다.
    rnn_300_test_loss, lstm_300_test_loss, gru_300_test_loss = main(300)

    print("=" * 20, "시계열 길이가 10 인 경우", "=" * 20)
    print("[RNN ] 테스트 MSE = {:.5f}".format(rnn_10_test_loss))
    print("[LSTM] 테스트 MSE = {:.5f}".format(lstm_10_test_loss))
    print("[GRU ] 테스트 MSE = {:.5f}".format(gru_10_test_loss))
    print()

    print("=" * 20, "시계열 길이가 300 인 경우", "=" * 20)
    print("[RNN ] 테스트 MSE = {:.5f}".format(rnn_300_test_loss))
    print("[LSTM] 테스트 MSE = {:.5f}".format(lstm_300_test_loss))
    print("[GRU ] 테스트 MSE = {:.5f}".format(gru_300_test_loss))
    print()


#@@ LSTM으로 IMDb 데이터 학습하기
from elice_utils import EliceUtils

elice_utils = EliceUtils()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data(num_words, max_len):
    # TODO: [지시사항 1번] IMDB 데이터셋을 불러오세요.
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words) #imdb의 loda_data함수를 통해 데이터 불러오고, 전체 단어의 갯수를 num_words로 정해줌. (d/t 자연어)
    X_train = pad_sequences(X_train, maxlen=max_len) #각 단어의 길이를 일괄적으로 맞춰주기 위해서 최대단어 개수를 max_len으로 맞춤
    X_test = pad_sequences(X_test, maxlen=max_len)

    return X_train, X_test, y_train, y_test


def build_lstm_model(num_words, embedding_len):
    model = Sequential()

    # TODO: [지시사항 2번] LSTM 기반 모델을 구성하세요.
    model.add(layers.Embedding(num_words, embedding_len)) #전체단어 갯수에서 각 embedding vector의 길이만큼 바꿈
    model.add(layers.LSTM(16))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def run_model(model, X_train, X_test, y_train, y_test, epochs=5):
    # TODO: [지시사항 3번] 모델 학습을 위한 optimizer, loss 함수, 평가 지표를 설정하세요.
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # TODO: [지시사항 4번] 모델 학습을 위한 hyperparameter를 설정하세요.
    hist = model.fit(X_train, y_train, batch_size=128, epochs=epochs, shuffle=True, verbose=2)

    # 모델을 테스트 데이터셋으로 테스트합니다.
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print()
    print("테스트 loss: {:.5f}, 테스트 정확도: {:.3f}%".format(test_loss, test_acc * 100))

    return optimizer, hist


def main():
    tf.random.set_seed(2022)

    num_words = 6000
    max_len = 130
    embedding_len = 100

    X_train, X_test, y_train, y_test = load_data(num_words, max_len)

    model = build_lstm_model(num_words, embedding_len)
    run_model(model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()

#@@ GRU를 통한 항공 승객 수 분석
from elice_utils import EliceUtils

elice_utils = EliceUtils()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(window_size):
    raw_data = pd.read_csv("./airline-passengers.csv")
    raw_passengers = raw_data["Passengers"].to_numpy()

    # 데이터의 평균과 표준편차 값으로 정규화(표준화) 합니다.
    mean_passenger = raw_passengers.mean()
    stdv_passenger = raw_passengers.std(ddof=0)
    raw_passengers = (raw_passengers - mean_passenger) / stdv_passenger
    plot_data = {"month": raw_data["Month"], "mean": mean_passenger, "stdv": stdv_passenger}

    # window_size 개의 데이터를 불러와 입력 데이터(X)로 설정하고
    # window_size보다 한 시점 뒤의 데이터를 예측할 대상(y)으로 설정하여
    # 데이터셋을 구성합니다.
    X, y = [], []
    for i in range(len(raw_passengers) - window_size):
        cur_passenger = raw_passengers[i:i + window_size]
        target = raw_passengers[i + window_size]

        X.append(list(cur_passenger))
        y.append(target)

    # X와 y를 numpy array로 변환합니다.
    X = np.array(X)
    y = np.array(y)

    # 각 입력 데이터는 sequence 길이가 window_size이고, featuer 개수는 1개가 되도록
    # 마지막에 새로운 차원을 추가합니다.
    # 즉, (전체 데이터 개수, window_size) -> (전체 데이터 개수, window_size, 1)이 되도록 변환합니다.
    X = X[:, :, np.newaxis]

    # 학습 데이터는 전체 데이터의 80%, 테스트 데이터는 20%로 설정합니다.
    total_len = len(X)
    train_len = int(total_len * 0.8)

    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    return X_train, X_test, y_train, y_test, plot_data


def build_gru_model(window_size):
    model = Sequential()

    # TODO: [지시사항 1번] GRU 기반 모델을 구성하세요.
    model.add(layers.GRU(4, input_shape=(window_size, 1)))
    model.add(layers.Dense(1))

    return model


def build_rnn_model(window_size):
    model = Sequential()

    # TODO: [지시사항 2번] SimpleRNN 기반 모델을 구성하세요.
    model.add(layers.SimpleRNN(4, input_shape=(window_size, 1)))
    model.add(layers.Dense(1))
    return model


def run_model(model, X_train, X_test, y_train, y_test, epochs=100, name=None):
    # TODO: [지시사항 3번] 모델 학습을 위한 optimizer와 loss 함수를 설정하세요.
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    # TODO: [지시사항 4번] 모델 학습을 위한 hyperparameter를 설정하세요.
    hist = model.fit(X_train,y_train, batch_size=8,  epochs=epochs, shuffle=True, verbose=2)

    # 테스트 데이터셋으로 모델을 테스트합니다.
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print()
    print("테스트 MSE: {:.5f}".format(test_loss))
    print()

    return optimizer, hist


def plot_result(model, X_true, y_true, plot_data, name):
    y_pred = model.predict(X_true)

    # 표준화된 결과를 다시 원래 값으로 변환합니다.
    y_true_orig = (y_true * plot_data["stdv"]) + plot_data["mean"]
    y_pred_orig = (y_pred * plot_data["stdv"]) + plot_data["mean"]

    # 테스트 데이터에서 사용한 날짜들만 가져옵니다.
    test_month = plot_data["month"][-len(y_true):]

    # 모델의 예측값을 실제값과 함께 그래프로 그립니다.
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.plot(y_true_orig, color="b", label="True")
    ax.plot(y_pred_orig, color="r", label="Prediction")
    ax.set_xticks(list(range(len(test_month))))
    ax.set_xticklabels(test_month, rotation=45)
    ax.set_title("{} Result".format(name))
    ax.legend(loc="upper left")
    plt.savefig("airline_{}.png".format(name.lower()))


def main():
    tf.random.set_seed(2022)

    window_size = 4
    X_train, X_test, y_train, y_test, plot_data = load_data(window_size)

    gru_model = build_gru_model(window_size)
    run_model(gru_model, X_train, X_test, y_train, y_test, name="GRU")
    plot_result(gru_model, X_test, y_test, plot_data, name="GRU")

    rnn_model = build_rnn_model(window_size)
    run_model(rnn_model, X_train, X_test, y_train, y_test, name="RNN")
    plot_result(rnn_model, X_test, y_test, plot_data, name="RNN")

    elice_utils.send_image("airline_{}.png".format("gru"))
    elice_utils.send_image("airline_{}.png".format("rnn"))


if __name__ == "__main__":
    main()

#@@ RNN기반 모델을 통한 분류 작업
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import pandas as pd

def load_data(max_len):
    data = pd.read_csv("./review_score.csv")
    # 리뷰 문장을 입력 데이터로, 해당 리뷰의 평점을 라벨 데이터로 설정합니다.
    X = data['Review']
    y = data['Score']
    y = y - 1 # 값을 1~5에서 0~4로 변경

    # 문장 내 각 단어를 숫자로 변환하는 Tokenizer를 적용합니다.
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)

    # 전체 단어 중에서 가장 큰 숫자로 mapping된 단어의 숫자를 가져옵니다.
    # 즉, max_features는 전체 데이터셋에 등장하는 겹치지 않는 단어의 개수 + 1과 동일합니다.
    max_features = max([max(_in) for _in in X]) + 1

    # 불러온 데이터셋을 학습 데이터 80%, 테스트 데이터 20%로 분리합니다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 모든 문장들을 가장 긴 문장의 단어 개수가 되게 padding을 추가합니다.
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    return X_train, X_test, y_train, y_test, max_features

def build_rnn_model(max_features, embedding_size):
    model = Sequential()

    # TODO: [지시사항 1번] Simple RNN 기반의 모델을 완성하세요.
    model.add(layers.Embedding(max_features, embedding_size))
    model.add(layers.SimpleRNN(20))
    model.add(layers.Dense(5, activation='softmax'))

    return model

def build_lstm_model(max_features, embedding_size):
    model = Sequential()

    # TODO: [지시사항 2번] LSTM 기반의 모델을 완성하세요.
    model.add(layers.Embedding(max_features, embedding_size))
    model.add(layers.LSTM(20))
    model.add(layers.Dense(5, activation='softmax'))

    return model

def build_gru_model(max_features, embedding_size):
    model = Sequential()

    # TODO: [지시사항 3번] GRU 기반의 모델을 완성하세요.
    model.add(layers.Embedding(max_features, embedding_size))
    model.add(layers.GRU(20))
    model.add(layers.Dense(5, activation='softmax'))

    return model

def run_model(model, X_train, X_test, y_train, y_test, epochs=10):
    # TODO: [지시사항 4번] 모델 학습을 위한 optimizer, loss 함수, 평가 지표를 설정하세요.
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TODO: [지시사항 5번] 모델 학습을 위한 hyperparameter를 설정하세요.
    hist = model.fit(X_train, y_train, batch_size=256, epochs=epochs, shuffle=True, verbose=2)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    return test_loss, test_acc, optimizer, hist

def main():
    tf.random.set_seed(2022)
    max_len = 150
    embedding_size = 128

    X_train, X_test, y_train, y_test, max_features = load_data(max_len)
    rnn_model = build_rnn_model(max_features, embedding_size)
    lstm_model = build_lstm_model(max_features, embedding_size)
    gru_model = build_gru_model(max_features, embedding_size)

    rnn_test_loss, rnn_test_acc, _, _ = run_model(rnn_model, X_train, X_test, y_train, y_test)
    lstm_test_loss, lstm_test_acc, _, _ = run_model(lstm_model, X_train, X_test, y_train, y_test)
    gru_test_loss, gru_test_acc, _, _ = run_model(gru_model, X_train, X_test, y_train, y_test)

    print()
    print("=" * 20, "모델 별 Test Loss와 정확도", "=" * 20)
    print("[RNN ] 테스트 Loss: {:.5f}, 테스트 Accuracy: {:.3f}%".format(rnn_test_loss, rnn_test_acc * 100))
    print("[LSTM] 테스트 Loss: {:.5f}, 테스트 Accuracy: {:.3f}%".format(lstm_test_loss, lstm_test_acc * 100))
    print("[GRU ] 테스트 Loss: {:.5f}, 테스트 Accuracy: {:.3f}%".format(gru_test_loss, gru_test_acc * 100))

if __name__ == "__main__":
    main()
#@@ RNN 기반 모델을 통한 회귀 분석
from elice_utils import EliceUtils

elice_utils = EliceUtils()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_data(window_size):
    raw_data_df = pd.read_csv("./AAPL.csv", index_col="Date") #"Date"의 칼럼을 인덱스로 사용하기로 setting

    # 데이터 전체를 표준화합니다.
    scaler = StandardScaler()
    raw_data = scaler.fit_transform(raw_data_df)
    plot_data = {"mean": scaler.mean_[3], "var": scaler.var_[3], "date": raw_data_df.index}

    # 입력 데이터(X)는 시작가, 일 최고가, 일 최저가, 종가 데이터를 사용하고
    # 라벨 데이터(y)는 4번째 컬럼에 해당하는 종가 데이터만 사용합니다.
    raw_X = raw_data[:, :4]
    raw_y = raw_data[:, 3]

    # window_size 개의 데이터를 불러와 입력 데이터(X)로 설정하고
    # window_size보다 한 시점 뒤의 데이터를 예측할 대상(y)으로 설정하여
    # 데이터셋을 구성합니다.
    X, y = [], []
    for i in range(len(raw_X) - window_size):
        cur_prices = raw_X[i:i + window_size, :]
        target = raw_y[i + window_size]

        X.append(list(cur_prices))
        y.append(target)

    # X와 y를 numpy array로 변환합니다.
    X = np.array(X)
    y = np.array(y)

    # 학습 데이터는 전체 데이터의 80%, 테스트 데이터는 20%로 설정합니다.
    total_len = len(X)
    train_len = int(total_len * 0.8)

    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    return X_train, X_test, y_train, y_test, plot_data


def build_rnn_model(window_size, num_features):
    model = Sequential()

    # TODO: [지시사항 1번] SimpleRNN 기반 모델을 구성하세요.
    model.add(layers.SimpleRNN(256, input_shape=(window_size, num_features)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))

    return model


def build_lstm_model(window_size, num_features):
    model = Sequential()

    # TODO: [지시사항 2번] LSTM 기반 모델을 구성하세요.
    model.add(layers.LSTM(256, input_shape=(window_size, num_features)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    return model


def build_gru_model(window_size, num_features):
    model = Sequential()

    # TODO: [지시사항 3번] GRU 기반 모델을 구성하세요.
    model.add(layers.GRU(256, input_shape=(window_size, num_features)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))

    return model


def run_model(model, X_train, X_test, y_train, y_test, epochs=10, name=None):
    # TODO: [지시사항 4번] 모델 학습을 위한 optimizer와 loss 함수를 설정하세요.
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    # TODO: [지시사항 5번] 모델 학습을 위한 hyperparameter를 설정하세요.
    hist = model.fit(X_train, y_train, batch_size=128, epochs=epochs, shuffle=True, verbose=2)

    # 테스트 데이터셋으로 모델을 테스트합니다.
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print("[{}] 테스트 loss: {:.5f}".format(name, test_loss))
    print()

    return optimizer, hist


def plot_result(model, X_true, y_true, plot_data, name):
    y_pred = model.predict(X_true)

    # 표준화된 결과를 다시 원래 값으로 변환합니다.
    y_true_orig = (y_true * np.sqrt(plot_data["var"])) + plot_data["mean"]
    y_pred_orig = (y_pred * np.sqrt(plot_data["var"])) + plot_data["mean"]

    # 테스트 데이터에서 사용한 날짜들만 가져옵니다.
    test_date = plot_data["date"][-len(y_true):]

    # 모델의 예측값을 실제값과 함께 그래프로 그립니다.
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.plot(y_true_orig, color="b", label="True")
    ax.plot(y_pred_orig, color="r", label="Prediction")
    ax.set_xticks(list(range(len(test_date))))
    ax.set_xticklabels(test_date, rotation=45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.set_title("{} Result".format(name))
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("apple_stock_{}".format(name.lower()))

    elice_utils.send_image("apple_stock_{}.png".format(name.lower()))


def main():
    tf.random.set_seed(2022)

    window_size = 30
    X_train, X_test, y_train, y_test, plot_data = load_data(window_size)
    num_features = X_train[0].shape[1]

    rnn_model = build_rnn_model(window_size, num_features)
    lstm_model = build_lstm_model(window_size, num_features)
    gru_model = build_gru_model(window_size, num_features)

    run_model(rnn_model, X_train, X_test, y_train, y_test, name="RNN")
    run_model(lstm_model, X_train, X_test, y_train, y_test, name="LSTM")
    run_model(gru_model, X_train, X_test, y_train, y_test, name="GRU")

    plot_result(rnn_model, X_test, y_test, plot_data, name="RNN")
    plot_result(lstm_model, X_test, y_test, plot_data, name="LSTM")
    plot_result(gru_model, X_test, y_test, plot_data, name="GRU")


if __name__ == "__main__":
    main()


#@@ test :CNN 모델로 CIFAR-10 분류하기
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from elice_utils import EliceUtils

elice_utils = EliceUtils()

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt


def load_cifar10():
    # CIFAR-10 데이터셋을 불러옵니다.
    X_train = np.load("cifar10_train_X.npy")
    y_train = np.load("cifar10_train_y.npy")
    X_test = np.load("cifar10_test_X.npy")
    y_test = np.load("cifar10_test_y.npy")

    # TODO: [지시사항 1번] 이미지의 각 픽셀값을 0에서 1 사이로 정규화하세요.
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 정수 형태로 이루어진 라벨 데이터를 one-hot encoding으로 바꿉니다.
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return X_train, X_test, y_train, y_test


def build_cnn_model(num_classes, input_shape):
    model = Sequential()

    # TODO: [지시사항 2번] 지시사항 대로 CNN 모델을 만드세요.
    model.add(layers.Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(2))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(2))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(2))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def plot_loss(hist):
    # hist 객체에서 train loss와 valid loss를 불러옵니다.
    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xticks(list(epochs))

    # ax를 이용하여 train loss와 valid loss를 plot 합니다..
    ax.plot(epochs, train_loss, marker=".", c="blue", label="Train Loss")
    ax.plot(epochs, val_loss, marker=".", c="red", label="Valid Loss")

    ax.legend(loc="upper right")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    fig.savefig("loss.png")


def plot_accuracy(hist):
    # hist 객체에서 train accuracy와 valid accuracy를 불러옵니다..
    train_acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]
    epochs = np.arange(1, len(train_acc) + 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xticks(list(epochs))
    # ax를 이용하여 train accuracy와와 valid accuracy와를 plot 합니다.
    ax.plot(epochs, val_acc, marker=".", c="red", label="Valid Accuracy")
    ax.plot(epochs, train_acc, marker=".", c="blue", label="Train Accuracy")

    ax.legend(loc="lower right")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    fig.savefig("accuracy.png")


def get_topk_accuracy(y_test, y_pred, k=1):
    # one-hot encoding으로 이루어진(y_test를 다시 정수 라벨 형식으로 바꿉니다.
    true_labels = np.argmax(y_test, axis=1)

    # y_pred를 확률값이 작은 것에서 큰 순서로 정렬합니다.
    pred_labels = np.argsort(y_pred, axis=1)

    correct = 0
    for true_label, pred_label in zip(true_labels, pred_labels):
        # TODO: [지시사항 3번] 현재 pred_label에서 확률값이 가장 큰 라벨 k개를 가져오세요
        cur_preds = None

        if true_label in cur_preds:
            correct += 1

    # TODO: [지시사항 3번] Top-k accuarcy를 구하세요.
    topk_accuracy = None

    return topk_accuracy


def main(model=None, epochs=5):
    # 시드 고정을 위한 코드입니다. 수정하지 마세요!
    tf.random.set_seed(2022)

    X_train, X_test, y_train, y_test = load_cifar10()
    cnn_model = build_cnn_model(len(y_train[0]), X_train[0].shape)
    cnn_model.summary()

    # TODO: [지시사항 4번] 지시사항 대로 모델의 optimizer, loss, metrics을 설정하세요.
    optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # TODO: [지시사항 5번] 지시사항 대로 hyperparameter를 설정하여 모델을 학습하세요.
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, shuffle=True, verbose=2)

    # Test 데이터를 적용했을 때 예측 확률을 구합니다.
    y_pred = cnn_model.predict(X_test)
    top1_accuracy = get_topk_accuracy(y_test, y_pred)
    top3_accuracy = get_topk_accuracy(y_test, y_pred, k=3)

    print("Top-1 Accuracy: {:.3f}%".format(top1_accuracy * 100))
    print("Top-3 Accuracy: {:.3f}%".format(top3_accuracy * 100))

    # Test accuracy를 구합니다.
    _, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)

    # Tensorflow로 구한 test accuracy와 top1 accuracy는 같아야 합니다.
    # 다만 부동 소수점 처리 문제로 완전히 같은 값이 나오지 않는 경우도 있어서
    # 소수점 셋째 자리까지 반올림하여 비교합니다.
    assert round(test_accuracy, 3) == round(top1_accuracy, 3)

    plot_loss(hist)
    plot_accuracy(hist)

    return optimizer, hist


if __name__ == '__main__':
    main()

#@@ 트위터 감정 분석
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_csv("./train.csv")

    # 트윗 문장과 해당 트윗의 감정 라벨을 불러옵니다.
    tweets = df["Tweet"]
    label = df["Label"]

    # 전체 라벨 개수를 가져옵니다.
    num_classes = len(pd.unique(label))

    # TODO: [지시사항 1번] 문장 내 각 단어를 숫자로 변환하는 Tokenizer를 적용하세요.
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    tweets = tokenizer.texts_to_sequences(tweets)

    # 전체 중복되지 않는 단어의 개수를 가져옵니다.
    num_words = max([max(tweet) for tweet in tweets]) + 1

    # 문장 중 가장 긴 문장에 있는 단어 개수를 가져옵니다.
    maxlen = 0
    for tweet in tweets:
        if len(tweet) > maxlen:
            maxlen = len(tweet)

    # TODO: [지시사항 1번] 불러온 데이터셋을 학습 데이터 80%, 테스트 데이터 20%로 분리하세요.
    X_train, X_test, y_train, y_test = train_test_split(tweets, label, test_size=0.2, random_state=2022)

    # 모든 문장들을 가장 긴 문장의 단어 개수가 되게 padding을 추가합니다.
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    return X_train, X_test, y_train, y_test, num_words, num_classes


def build_lstm_model(num_words, embedding_len, num_classes):
    model = Sequential()

    # TODO: [지시사항 2번] LSTM 기반 모델을 완성하세요.
    model.add(layers.Embedding(num_words, embedding_len))
    model.add(layers.LSTM(300))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def main(model=None, epochs=10):
    tf.random.set_seed(2022)

    embedding_len = 50
    X_train, X_test, y_train, y_test, num_words, num_classes = load_data()

    model = build_lstm_model(num_words, embedding_len, num_classes)

    # TODO: [지시사항 3번] Optimizer, Loss 함수, Metrics과 모델 학습을 위한 hyperparameter를 완성하세요.
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # 모델 컴파일
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=32, shuffle=True, verbose=2)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("테스트 Loss: {:.5f}, 테스트 정확도: {:.3f}%".format(test_loss, test_acc * 100))

    return optimizer, hist


if __name__ == "__main__":
    main()