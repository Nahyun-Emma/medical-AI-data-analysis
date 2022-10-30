#@@ perceptron 작동 예시 구현하기
'''
1. 신호의 총합과 외출 여부를 반환하는
   Perceptron 함수를 완성합니다.

   Step01. Bias는 외출을 좋아하는 정도이며
           -1로 설정되어 있습니다.

   Step02. 입력 받은 값과 Bias 값을 이용하여 신호의
           총합을 구합니다.

   Step03. 지시한 활성화 함수를 참고하여 외출 여부
           (0 or 1)를 반환합니다.
'''


def Perceptron(x_1, x_2, w_1, w_2):
    bias = -1

    output = w_1 * x_1 + w_2 * x_2 + bias

    y = 1 if output > 0 else 0

    return output, y


# 값을 입력받는 함수입니다.

def input_func():
    # 비 오는 여부(비가 온다 : 1 / 비가 오지 않는다 : 0)
    x_1 = int(input("x_1 : 비가 오는 여부(1 or 0)을 입력하세요."))

    # 여자친구가 만나자고 하는 여부(만나자고 한다 : 1 / 만나자고 하지 않는다 : 0)
    x_2 = int(input("x_2 : 여친이 만나자고 하는 여부(1 or 0)을 입력하세요."))

    # 비를 좋아하는 정도의 값(비를 싫어한다 -5 ~ 5 비를 좋아한다)
    w_1 = int(input("w_1 : 비를 좋아하는 정도 값(-5 ~ 5)을 입력하세요."))

    # 여자친구를 좋아하는 정도의 값(여자친구를 싫어한다 -5 ~ 5 비를 좋아한다)
    w_2 = int(input("w_2 : 여친을 좋아하는 정도 값(-5 ~ 5)을 입력하세요."))

    return x_1, x_2, w_1, w_2


'''
2. 실행 버튼을 눌러 x1, x2, w1, w2 값을 다양하게
   입력해보고, Perceptron함수에서 반환한 신호의 총합과
   그에 따른 외출 여부를 확인해보세요
'''


def main():
    x_1, x_2, w_1, w_2 = input_func()

    result, go_out = Perceptron(x_1, x_2, w_1, w_2)

    print("\n신호의 총합 : %d" % result)

    if go_out > 0:
        print("외출 여부 : %d\n ==> 외출한다!" % go_out)
    else:
        print("외출 여부 : %d\n ==> 외출하지 않는다!" % go_out)


if __name__ == "__main__":
    main()


#@@ DIY 퍼셉트론 만들기
'''
1. 가중치와 Bias 값을 
   임의로 설정해줍니다.

   Step01. 0이상 1미만의 임의의 값으로 정의된 
           4개의 가중치 값이 들어가있는 
           1차원 리스트를 정의해줍니다.

   Step02. Bias 값을 임의의 값으로 설정해줍니다.
'''


def main():
    x = [1, 2, 3, 4]

    w = [0.3, 0.5, 0.1, 0.7]  # 임의로 셋팅
    b = -2  # 임의로 셋팅

    output, y = perceptron(w, x, b)

    print('output: ', output)
    print('y: ', y)


'''
2. 신호의 총합과 그에 따른 결과 0 또는 1을
   반환하는 함수 perceptron을 완성합니다.

   Step01. 입력 받은 값과 Bias 값을 이용하여
           신호의 총합을 구합니다.

   Step02. 신호의 총합이 0 이상이면 1을, 
           그렇지 않으면 0을 반환하는 활성화 
           함수를 작성합니다.
'''


def perceptron(w, x, b):
    output = x[0] * w[0] + x[1] * w[1] + x[2] * w[2] + x[3] * w[3] + b

    y = 1 if output >= 0 else 0

    return output, y


if __name__ == "__main__":
    main()


#@@ 비선형적인 문제 : XOR 문제
import numpy as np

'''
1. XOR_gate 함수를 최대한 완성해보세요.

   Step01. 이전 실습을 참고하여 입력값 x1과 x2를
           Numpy array 형식으로 정의한 후, x1과 x2에
           각각 곱해줄 가중치도 Numpy array 형식으로 
           적절히 설정해주세요.

   Step02. XOR_gate를 만족하는 Bias 값을
           적절히 설정해주세요.

   Step03. 가중치, 입력값, Bias를 이용하여 
           가중 신호의 총합을 구합니다.

   Step04. Step Function 함수를 호출하여 
           XOR_gate 출력값을 반환합니다.
'''


def XOR_gate(x1, x2):
    x = np.array([x1, x2])

    weight = np.array([0.5, -0.5])

    bias = 0.1

    y = np.matmul(x, weight) + bias

    return Step_Function(y)


'''
2. 설명을 보고 Step Function을 완성합니다.
   앞 실습에서 구현한 함수를 그대로 
   사용할 수 있습니다.

   Step01. 0 미만의 값이 들어오면 0을,
           0 이상의 값이 들어오면 1을
           출력하는 함수를 구현하면 됩니다.
'''


def Step_Function(y):
    return 0 if y > 0 else 1


def main():
    # XOR Gate에 넣어줄 Input과 그에 따른 Output
    Input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Output = np.array([[0], [1], [1], [0]])

    # XOR Gate를 만족하는지 출력하여 확인
    print('XOR Gate 출력')

    XOR_list = []

    for x1, x2 in Input:
        print('Input: ', x1, x2, ' Output: ', XOR_gate(x1, x2))
        XOR_list.append(XOR_gate(x1, x2))

    hit = 0
    for i in range(len(Output)):
        if XOR_list[i] == Output[i]:
            hit += 1

    acc = float(hit / 4) * 100

    print('Accuracy: %.1lf%%' % (acc))


if __name__ == "__main__":
    main()

#@@ 다층 퍼셉트론으로 XOR gate 구현하기
import numpy as np

'''
1. AND_gate 함수를 완성하세요. 
'''


def AND_gate(x1, x2):
    x = np.array([x1, x2])

    weight = np.array([0.5, 0.5])

    bias = -0.7

    y = np.matmul(x, weight) + bias

    return Step_Function(y)


'''
2. OR_gate 함수를 완성하세요.
'''


def OR_gate(x1, x2):
    x = np.array([x1, x2])

    weight = np.array([0.5, 0.5])

    bias = -0.3

    y = np.matmul(x, weight) + bias

    return Step_Function(y)


'''
3. NAND_gate 함수를 완성하세요.
'''


def NAND_gate(x1, x2):
    x = np.array([x1, x2])

    weight = np.array([-0.5, -0.5])

    bias = 0.7

    y = np.matmul(x, weight) + bias

    return Step_Function(y)


'''
4. Step_Function 함수를 완성하세요.
'''


def Step_Function(y):
    return 1 if y >= 0 else 0


'''
5. AND_gate, OR_gate, NAND_gate 함수들을
   활용하여 XOR_gate 함수를 완성하세요. 앞서 만든
   함수를 활용하여 반환되는 값을 정의하세요.
'''


def XOR_gate(x1, x2):
    nand_out = NAND_gate(x1, x2)
    or_out = OR_gate(x1, x2)
    and_out = AND_gate(nand_out, or_out)

    return and_out


def main():
    # XOR gate에 넣어줄 Input
    array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # XOR gate를 만족하는지 출력하여 확인
    print('XOR Gate 출력')

    for x1, x2 in array:
        print('Input: ', x1, x2, ', Output: ', XOR_gate(x1, x2))


if __name__ == "__main__":
    main()

#@@ MLP(다층퍼셉트론) 모델로 2D 데이터 분류하기
import numpy as np
from visual import *
from sklearn.neural_network import MLPClassifier

from elice_utils import EliceUtils

elice_utils = EliceUtils()

import warnings

warnings.filterwarnings(action='ignore')

np.random.seed(100)


# 데이터를 읽어오는 함수입니다.

def read_data(filename):
    X = []
    Y = []

    with open(filename) as fp:
        N, M = fp.readline().split()
        N = int(N)
        M = int(M)

        for i in range(N):
            line = fp.readline().split()
            for j in range(M):
                X.append([i, j])
                Y.append(int(line[j]))

    X = np.array(X)
    Y = np.array(Y)

    return (X, Y)


'''
1. MLPClassifier를 정의하고 hidden_layer_sizes를
   조정해 hidden layer의 크기 및 레이어의 개수를
   바꿔본 후, 학습을 시킵니다.
'''


def train_MLP_classifier(X, Y):
    clf = MLPClassifier(hidden_layer_sizes=(100, 100))
    clf.fit(X, Y)

    return clf


'''
2. 테스트 데이터에 대한 정확도를 출력하는 
   함수를 완성합니다. 설명을 보고 score의 코드를
   작성해보세요.
'''


def report_clf_stats(clf, X, Y):
    hit = 0  # 맞춘 갯수 담을 그릇
    miss = 0  # 틀린갯수 담을 그릇

    for x, y in zip(X, Y):
        if clf.predict([x])[0] == y:
            hit += 1
        else:
            miss += 1

    score = hit / len(x)

    print("Accuracy: %.1lf%% (%d hit / %d miss)" % (score, hit, miss))


'''
3. main 함수를 완성합니다.

   Step01. 학습용 데이터인 X_train, Y_train과
           테스트용 데이터인 X_test, Y_test를 각각
           read_data에서 반환한 값으로 정의합니다. 

           우리가 사용할 train.txt 데이터셋과
           test.txt 데이터셋은 data 폴더에 위치합니다.

   Step02. 앞에서 학습시킨 다층 퍼셉트론 분류 
           모델을 'clf'로 정의합니다. 'clf'의 변수로는
           X_train과 Y_train으로 설정합니다.

   Step03. 앞에서 완성한 정확도 출력 함수를
           'score'로 정의합니다. 'score'의 변수로는
           X_test와 Y_test로 설정합니다.
'''


def main():
    X_train, Y_train = read_data('data/train.txt')

    X_test, Y_test = read_data('data/test.txt')

    clf = train_MLP_classifier(X_train, Y_train)

    score = report_clf_stats(clf, X_test, Y_test)

    visualize(clf, X_test, Y_test)


if __name__ == "__main__":
    main()

#@@ 퍼셉트론 선형분류기를 이용해 붓꽃 데이터 분류하기
import numpy as np

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

from elice_utils import EliceUtils

elice_utils = EliceUtils()

np.random.seed(100)

'''
1. iris 데이터를 불러오고, 
   불러온 데이터를 학습용, 테스트용 데이터로 
   분리하여 반환하는 함수를 구현합니다.

   Step01. 불러온 데이터를 학습용 데이터 80%, 
           테스트용 데이터 20%로 분리합니다.

           일관된 결과 확인을 위해 random_state를 
           0으로 설정합니다.        
'''


def load_data():
    iris = load_iris()

    X = iris.data[:, 2:4]
    Y = iris.target

    X_train, X_test, Y_train, Y_test = None

    return X_train, X_test, Y_train, Y_test


'''
2. 사이킷런의 Perceptron 클래스를 사용하여 
   Perceptron 모델을 정의하고,
   학습용 데이터에 대해 학습시킵니다.

   Step01. 앞서 완성한 함수를 통해 데이터를 불러옵니다.

   Step02. Perceptron 모델을 정의합니다.
           max_iter와 eta0를 자유롭게 설정해보세요.

   Step03. 학습용 데이터에 대해 모델을 학습시킵니다.

   Step04. 테스트 데이터에 대한 모델 예측을 수행합니다. 
'''


def main():
    X_train, X_test, Y_train, Y_test = None

    perceptron = None

    None

    pred = None

    accuracy = accuracy_score(pred, Y_test)

    print("Test 데이터에 대한 정확도 : %0.5f" % accuracy)

    return X_train, X_test, Y_train, Y_test, pred


if __name__ == "__main__":
    main()import numpy as np

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

from elice_utils import EliceUtils
elice_utils = EliceUtils()

np.random.seed(100)

'''
1. iris 데이터를 불러오고, 
   불러온 데이터를 학습용, 테스트용 데이터로 
   분리하여 반환하는 함수를 구현합니다.
   
   Step01. 불러온 데이터를 학습용 데이터 80%, 
           테스트용 데이터 20%로 분리합니다.
           
           일관된 결과 확인을 위해 random_state를 
           0으로 설정합니다.        
'''

def load_data():

    iris = load_iris()

    X = iris.data[:,2:4]
    Y = iris.target

    X_train, X_test, Y_train, Y_test = None

    return X_train, X_test, Y_train, Y_test

'''
2. 사이킷런의 Perceptron 클래스를 사용하여 
   Perceptron 모델을 정의하고,
   학습용 데이터에 대해 학습시킵니다.
   
   Step01. 앞서 완성한 함수를 통해 데이터를 불러옵니다.
   
   Step02. Perceptron 모델을 정의합니다.
           max_iter와 eta0를 자유롭게 설정해보세요.
   
   Step03. 학습용 데이터에 대해 모델을 학습시킵니다.
   
   Step04. 테스트 데이터에 대한 모델 예측을 수행합니다. 
'''

def main():

    X_train, X_test, Y_train, Y_test = None

    perceptron = None

    None

    pred = None

    accuracy = accuracy_score(pred, Y_test)

    print("Test 데이터에 대한 정확도 : %0.5f" % accuracy)

    return X_train, X_test, Y_train, Y_test, pred

if __name__ == "__main__":
    main()

#@@ Gradient descent 알고리즘 구현하기
import numpy as np


# 사용할 1차 선형 회귀 모델

def linear_model(w0, w1, X):
    f_x = w0 + w1 * X

    return f_x


'''
1. 설명 중 '손실 함수' 파트의 수식을 참고해
   MSE 손실 함수를 완성하세요. 
'''


def Loss(f_x, y):
    ls = np.mean(np.square(y - f_x))

    return ls


'''
2. 설명 중 'Gradient' 파트의 마지막 두 수식을 참고해 두 가중치
   w0와 w1에 대한 gradient인 'gradient0'와 'gradient1'을
   반환하는 함수 gradient_descent 함수를 완성하세요.

   Step01. w0에 대한 gradient인 'gradient0'를 작성합니다.

   Step02. w1에 대한 gradient인 'gradient1'을 작성합니다.
'''


def gradient_descent(w0, w1, X, y):
    gradient0 = 2 * np.mean((y - (w0 + w1 * X)) * (-1))
    gradient1 = 2 * np.mean((y - (w0 + w1 * X)) * (-1 * X))

    return np.array([gradient0, gradient1])


'''
3. 설명 중 '가중치 업데이트' 파트의 두 수식을 참고해 
   gradient descent를 통한 가중치 업데이트 코드를 작성하세요.

   Step01. 앞서 완성한 gradient_descent 함수를 이용해
           w0와 w1에 대한 gradient인 'gd'를 정의하세요.

   Step02. 변수 'w0'와 'w1'에 두 가중치 w0와 w1을 
           업데이트하는 코드를 작성합니다. 앞서 정의한
           변수 'gd'와 이미 정의된 변수 'lr'을 사용하세요.
'''


def main():
    X = np.array([1, 2, 3, 4]).reshape((-1, 1))  # input vector
    y = np.array([3.1, 4.9, 7.2, 8.9]).reshape((-1, 1))  # answer vector

    # 파라미터 초기화
    w0 = 0
    w1 = 0

    # learning rate 설정
    lr = 0.001

    # 반복 횟수 1000으로 설정
    for i in range(1000):

        gd = gradient_descent(w0, w1, X, y)

        w0 = w0 - lr * gd[0]
        w1 = w1 - lr * gd[1]

        # 100회마다의 해당 loss와 w0, w1 출력
        if (i % 100 == 0):
            loss = Loss(linear_model(w0, w1, X), y)

            print("{}번째 loss : {}".format(i, loss))
            print("{}번째 w0, w1 : {}, {}".format(i, w0, w1), '\n')

    return w0, w1


if __name__ == '__main__':
    main()


#@@ 역전파(back propagation)
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


'''
X, y 를 가장 잘 설명하는 parameter (w1, w2, w3)를 반환하는
함수를 작성하세요. 여기서 X는 (x1, x2, x3) 의 list이며, y 는
0 혹은 1로 이루어진 list입니다. 예를 들어, X, y는 다음의 값을
가질 수 있습니다.

    X = [(1, 0, 0), (1, 0, 1), (0, 0, 1)]
    y = [0, 1, 1]
'''

'''
1. 지시 사항을 따라서 getParameters 함수를 완성하세요.

Step01. X의 한 원소가 3개이므로 가중치도 3개가 있어야 합니다.
        초기 가중치 w를 [1,1,1]로 정의하는 코드를 작성하세요.

        단순히 f = 3, w = [1,1,1]이라고 하는 것보다 좀 더 
        좋은 표현을 생각해보세요.


Step02. 초기 가중치 w를 모델에 맞게 계속 업데이트 해야합니다.

        업데이트를 위해 초기 가중치 w에 더해지는 값들의 리스트
        wPrime을 [0,0,0]로 정의하는 코드를 작성하세요.  

        마찬가지로 단순히 wPrime = [0,0,0]이라고 하는 것보다
        좀 더 좋은 표현을 생각해보세요.


Step03. sigmoid 함수를 통과할 r값을 정의해야합니다. r은 
        X의 각 값과 그에 해당하는 가중치 w의 곱의 합입니다.

        즉, r = X_0_0 * w_0 + X_1_0 * w_0 + ... + X_2_2 * w_2
        가 됩니다.

        그리고 sigmoid 함수를 통과한 r값을 v로 정의합시다.


Step04. 가중치 w가 더이상 업데이트가 안될 때까지 업데이트 해줘야합니다.
        즉, 가중치 w의 업데이트를 위해 더해지는 wPrime의 절댓값이 어느 정도까지
        작아지면 업데이트를 끝내야 합니다. 

        그 값을 0.001로 정하고, wPrime이 그 값을 넘지 못하면 가중치 
        업데이트를 끝내도록 합시다. 

        다만 wPrime의 절댓값이 0.001보다 작아지기 전까지는 w에 wPrime을 계속
        더하면서 w를 업데이트 합시다.    
'''


def getParameters(X, y):
    # Step01.

    f = len(X[0])  # X의 첫번째 원소의 크기 (X의 원소의 갯수)

    w = [1] * f  #

    values = []

    while True:

        # Step02.

        wPrime = [0] * f  # 0으로 초기화
        vv = []  # sigmoid 통과한 r이 들어갈 빈 리스트
        # Step03.

        for i in range(len(y)):
            r = 0
            for j in range(f):
                r = r + X[i][j] * w[j]

            v = sigmoid(r)
            vv.append(v)
            # w를 업데이트하기 위한 wPrime을 역전파를 이용해 구하는 식
            for j in range(f):
                wPrime[j] += -((v - y[i]) * v * (1 - v) * X[i][j])

        # Step04. 업데이트 하는 과정

        flag = False

        for i in range(f):
            if abs(wPrime[i]) >= 0.001:  # wPrime의 절댓값이 0.001보다 클때는 계속 진행
                flag = True
                break

        if flag == False:
            break

        for j in range(f):
            w[j] = w[j] + wPrime[j]  # weight 업데이트 식

    return w


def main():
    '''
    이 코드는 수정하지 마세요.
    '''

    X = [(1, 0, 0), (1, 0, 1), (0, 0, 1)]
    y = [0, 1, 1]

    '''
    # 아래의 예제 또한 테스트 해보세요.
    X = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    y = [0, 0, 1, 1, 1, 1, 1, 1]

    # 아래의 예제를 perceptron이 100% training할 수 있는지도 확인해봅니다.
    X = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    y = [0, 0, 0, 1, 0, 1, 1, 1]
    '''

    print(getParameters(X, y))


if __name__ == "__main__":
    main()

#@@ 텐서플로우 버전 비교하기
import numpy as np
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""""
텐서플로우 1.x 버전
"""""


def tf1():
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    # 상수
    a = tf.constant(5)
    b = tf.constant(3)

    # 계산 정의
    add_op = a + b

    # 세션 시작
    sess = tf.Session()
    result_tf1 = sess.run(add_op)

    return a, b, result_tf1


"""""
텐서플로우 2.0 버전
"""""


def tf2():
    import tensorflow as tf
    tf.compat.v1.enable_v2_behavior()

    # 상수
    a = tf.constant(5)
    b = tf.constant(3)

    # 즉시 실행 연산
    result_tf2 = tf.add(a, b)

    return a, b, result_tf2.numpy()


def main():
    tf_2, tf_1 = tf2()[2], tf1()[2]

    print('result_tf1:', tf_1)
    print('result_tf2:', tf_2)


if __name__ == "__main__":
    main()

#@@ 텐서데이터 생성
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
1. 상수 텐서를 생성하는 constant_tensors 함수를 완성하세요.

   Step01. 5의 값을 가지는 (1,1) shape의 8-bit integer 텐서를 만드세요.

   Step02. 모든 원소의 값이 0인 (3,5) shape의 16-bit integer 텐서를 만드세요.

   Step03. 모든 원소의 값이 1인 (4,3) shape의 8-bit integer 텐서를 만드세요.
'''


def constant_tensors():
    t1 = tf.constant(5, shape=(1, 1), dtype=tf.int8)

    t2 = tf.zeros(shape=(3, 5), dtype=tf.int16)

    t3 = tf.ones(shape=(4, 3), dtype=tf.int8)

    return t1, t2, t3


'''
2. 시퀀스 텐서를 생성하는 sequence_tensors 함수를 완성하세요. 

   Step01. 1.5에서 10.5까지 증가하는 3개의 텐서를 만드세요.

   Step02. 2.5에서 20.5까지 증가하는 5개의 텐서를 만드세요. 
'''


def sequence_tensors():
    seq_t1 = tf.range(1.5, 11, 4.5)

    seq_t2 = tf.range(2.5, 21, 4.5)

    return seq_t1, seq_t2


'''
3. 변수를 생성하는 variable_tensor 함수를 완성하세요.

   Step01. 값이 100인 변수 텐서를 만드세요.

   Step02. 모든 원소의 값이 1인 (2,2) shape의 변수 텐서를 만드세요.
           이름도 'W'로 지정합니다.

   Step03. 모든 원소의 값이 0인 (2,) shape의 변수 텐서를 만드세요.
           이름도 'b'로 지정합니다.
'''


def variable_tensor():
    var_tensor = tf.Variable(initial_value=100)

    W = tf.Variable(tf.ones(shape=(2, 2), name="W"))

    b = tf.Variable(tf.zeros(shape=(2,), name="b"))

    return var_tensor, W, b


def main():
    t1, t2, t3 = constant_tensors()

    seq_t1, seq_t2 = sequence_tensors()

    var_tensor, W, b = variable_tensor()

    constant_dict = {'t1': t1, 't2': t2, 't3': t3}

    sequence_dict = {'seq_t1': seq_t1, 'seq_t2': seq_t2}

    variable_dict = {'var_tensor': var_tensor, 'W': W, 'b': b}

    for key, value in constant_dict.items():
        print(key, ' :', value.numpy())

    print()

    for key, value in sequence_dict.items():
        print(key, ' :', value.numpy())

    print()

    for key, value in variable_dict.items():
        print(key, ' :', value.numpy())


if __name__ == "__main__":
    main()

#@@ 텐서 연산
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
1. 이항 연산자를 사용해 사칙 연산을 수행하여 각 변수에 저장하세요.

   Step01. 텐서 'a'와 'b'를 더해 'add'에 저장하세요.

   Step02. 텐서 'a'에서 'b'를 빼 'sub'에 저장하세요.

   Step03. 텐서 'a'와 'b'를 곱해 'mul'에 저장하세요.

   Step04. 텐서 'a'에서 'b'를 나눠 'div'에 저장하세요.
'''


def main():
    a = tf.constant(10, dtype=tf.int32)
    b = tf.constant(3, dtype=tf.int32)

    add = tf.add(a, b)
    sub = tf.subtract(a, b)
    mul = tf.multiply(a, b)
    div = tf.truediv(a, b)

    tensor_dict = {'add': add, 'sub': sub, 'mul': mul, 'div': div}

    for key, value in tensor_dict.items():
        print(key, ' :', value.numpy(), '\n')

    return add, sub, mul, div


if __name__ == "__main__":
    main()

#@@ 텐서플로우 계산기 구현하기
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
1. 두 실수와 연산 종류를 입력받는 함수입니다. 코드를 살펴보세요.
'''


def insert():
    x = float(input('정수 또는 실수를 입력하세요. x : '))
    y = float(input('정수 또는 실수를 입력하세요. y : '))
    cal = input('어떤 연산을 할것인지 입력하세요. (+, -, *, /)')

    return x, y, cal


'''
2. 입력받는 연산의 종류 cal에 따라 연산을 수행하고
   결과를 반환하는 calcul() 함수를 완성하세요.
'''


def calcul(x, y, cal):
    result = 0

    # 더하기
    if cal == "+":
        result = tf.add(x, y)
    # 빼기
    elif cal == "-":
        result = tf.subtract(x, y)
    # 곱하기
    elif cal == "*":
        result = tf.multiply(x, y)
    # 나누기
    elif cal == "/":
        result = tf.divide(x, y)

    return result.numpy()


'''
3. 두 실수와 연산 종류를 입력받는 insert 함수를 호출합니다. 그 다음
   calcul 함수를 호출해 실수 사칙연산을 수행하고 결과를 출력합니다.
'''


def main():
    x, y, cal = insert()

    print(calcul(x, y, cal))


if __name__ == "__main__":
    main()

#@@ 텐서플로우를 활용하여 선형회귀 구현하기
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from elice_utils import EliceUtils

elice_utils = EliceUtils()

np.random.seed(100)

'''
1. 선형 회귀 모델의 클래스를 구현합니다.

   Step01. 가중치 초기값을 1.5의 값을 가진 변수 텐서로 설정하세요.

   Step02. Bias 초기값을 1.5의 값을 가진 변수 텐서로 설정하세요.

   Step03. W, X, b를 사용해 선형 모델을 구현하세요.
'''


class LinearModel:

    def __init__(self):
        self.W = tf.Variable(1.5)

        self.b = tf.Variable(1.5)

    def __call__(self, X, Y):
        return tf.add(tf.multiply(X, self.W), self.b)


'''
2. MSE 값을 계산해 반환하는 손실 함수를 완성합니다. 
'''


def loss(y, pred):
    return tf.reduce_mean(tf.square(y - pred))


'''
3. gradient descent 방식으로 학습하는 train 함수입니다.
   코드를 보면서 어떤 방식으로 W(가중치)와 b(Bias)이
   업데이트 되는지 확인해 보세요.
'''


def train(linear_model, x, y):
    with tf.GradientTape() as t:  # GradientTape : 연산에 대한기록을 다 저장시키는 함수
        current_loss = loss(y, linear_model(x, y))

    # learning_rate 값 선언
    learning_rate = 0.001

    # gradient 값 계산
    delta_W, delta_b = t.gradient(current_loss, [linear_model.W, linear_model.b])

    # learning rate와 계산한 gradient 값을 이용하여 업데이트할 파라미터 변화 값 계산
    W_update = (learning_rate * delta_W)
    b_update = (learning_rate * delta_b)

    return W_update, b_update


def main():
    # 데이터 생성
    x_data = np.linspace(0, 10, 50)
    y_data = 4 * x_data + np.random.randn(*x_data.shape) * 4 + 3

    # 데이터 출력
    plt.scatter(x_data, y_data)
    plt.savefig('data.png')
    elice_utils.send_image('data.png')

    # 선형 함수 적용
    linear_model = LinearModel()

    # epochs 값 선언
    epochs = 100

    # epoch 값만큼 모델 학습
    for epoch_count in range(epochs):

        # 선형 모델의 예측 값 저장
        y_pred_data = linear_model(x_data, y_data)

        # 예측 값과 실제 데이터 값과의 loss 함수 값 저장
        real_loss = loss(y_data, linear_model(x_data, y_data))

        # 현재의 선형 모델을 사용하여  loss 값을 줄이는 새로운 파라미터로 갱신할 파라미터 변화 값을 계산
        update_W, update_b = train(linear_model, x_data, y_data)

        # 선형 모델의 가중치와 Bias를 업데이트합니다.
        linear_model.W.assign_sub(update_W)
        linear_model.b.assign_sub(update_b)

        # 20번 마다 출력 (조건문 변경 가능)
        if (epoch_count % 20 == 0):
            print(f"Epoch count {epoch_count}: Loss value: {real_loss.numpy()}")
            print('W: {}, b: {}'.format(linear_model.W.numpy(), linear_model.b.numpy()))

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(x_data, y_data)
            ax1.plot(x_data, y_pred_data, color='red')
            plt.savefig('prediction.png')
            elice_utils.send_image('prediction.png')


if __name__ == "__main__":
    main()


#@@ 텐서플로우와 케라스를 활용하여 비선형회귀 구현하기
import tensorflow as tf
import numpy as np
from visual import *

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(100)
tf.random.set_seed(100)


def main():
    # 비선형 데이터 생성

    x_data = np.linspace(0, 10, 100)
    y_data = 1.5 * x_data ** 2 - 12 * x_data + np.random.randn(*x_data.shape) * 2 + 0.5

    '''
    1. 다층 퍼셉트론 모델을 만듭니다.
    '''

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(20, input_dim=1, activation='relu'),  # hidde node :20
        tf.keras.layers.Dense(20, activation='relu'),
        # 두번째 input_dim 설정 안해도 됨. 직전의 layer의 output dimension이 input으로 자동으로 설정됨.
        tf.keras.layers.Dense(1)
    ])  # 연속적으로 층을 쌓아 만드는 Sequential 모델을 위한 함수

    '''
    2. 모델 학습 방법을 설정합니다.
    '''

    model.compile(loss='mean_squared_error', optimizer='adam')  # 학습 방법 설정

    '''
    3. 모델을 학습시킵니다.
    '''

    history = model.fit(x_data, y_data, epochs=500, verbose=2)

    '''
    4. 학습된 모델을 사용하여 예측값 생성 및 저장
    '''

    predictions = model.predict(x_data)

    Visualize(x_data, y_data, predictions)

    return history, model


if __name__ == '__main__':
    main()

#@@  텐서플로우로 XOR 문제 해결하기
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # XOR 문제를 위한 데이터 생성

    training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
    target_data = np.array([[0], [1], [1], [0]], "float32")

    '''
    1. 다층 퍼셉트론 모델을 생성합니다.
    '''

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, input_dim=2, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    '''
    2. 모델의 손실 함수, 최적화 방법, 평가 방법을 설정합니다.
    '''

    model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'])

    '''
    3. 모델을 학습시킵니다. epochs를 자유롭게 설정해보세요.
    '''

    hist = model.fit(training_data, target_data, epochs=30)

    score = hist.history['binary_accuracy'][-1]

    print('최종 정확도: ', score * 100, '%')

    return hist


if __name__ == "__main__":
    main()

#@@ fashion-MNIST 데이터 분류하기
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import elice_utils

elice_utils = elice_utils.EliceUtils()

np.random.seed(100)
tf.random.set_seed(100)

'''
1. 다층 퍼셉트론 분류 모델을 만들고, 학습 방법을 설정해 
   학습시킨 모델을 반환하는 MLP 함수를 구현하세요.

   Step01. 다층 퍼셉트론 분류 모델을 생성합니다. 
           여러 층의 레이어를 쌓아 모델을 구성해보세요.

   Step02. 모델의 손실 함수, 최적화 방법, 평가 방법을 설정합니다.

   Step03. 모델을 학습시킵니다. epochs를 자유롭게 설정해보세요.
'''


def MLP(x_train, y_train):
    model = tf.keras.models.Sequential([None])

    model.compile(None)

    None

    return model


def main():
    x_train = np.loadtxt('./data/train_images.csv', delimiter=',', dtype=np.float32)
    y_train = np.loadtxt('./data/train_labels.csv', delimiter=',', dtype=np.float32)
    x_test = np.loadtxt('./data/test_images.csv', delimiter=',', dtype=np.float32)
    y_test = np.loadtxt('./data/test_labels.csv', delimiter=',', dtype=np.float32)

    # 이미지 데이터를 0~1범위의 값으로 바꾸어 줍니다.
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = MLP(x_train, y_train)

    # 학습한 모델을 test 데이터를 활용하여 평가합니다.
    loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print('\nTEST 정확도 :', test_acc)

    # 임의의 3가지 test data의 이미지와 레이블값을 출력하고 예측된 레이블값 출력
    predictions = model.predict(x_test)
    rand_n = np.random.randint(100, size=3)

    for i in rand_n:
        img = x_test[i].reshape(28, 28)
        plt.imshow(img, cmap="gray")
        plt.show()
        plt.savefig("test.png")
        elice_utils.send_image("test.png")

        print("Label: ", y_test[i])
        print("Prediction: ", np.argmax(predictions[i]))


if __name__ == "__main__":
    main()