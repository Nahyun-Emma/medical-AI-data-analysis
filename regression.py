#@@ loss function
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def loss(x, y, beta_0, beta_1):
    N = len(x) #반복문으로 작성할때 필요함
    '''

    loss = 0 
    for i in range(N):
        x_i = x[i]
        y_i = y[i]
        y_i_pd = beta_0*x  + beta_1
        loss += (y_i_pd - y_i)**2 
    return loss
    '''

    '''
    x, y, beta_0, beta_1 을 이용해 loss값을 계산한 뒤 리턴합니다.
    '''
    x = np.array(x)
    y = np.array(y)
    y_predict = beta_0 * x + beta_1
    total_loss = np.sum((y - y_predict) ** 2)
    return total_loss


X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513,
     5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441,
     5.19692852]

beta_0 = 1  # 기울기
beta_1 = 0.5  # 절편

print("Loss: %f" % loss(X, Y, beta_0, beta_1))

plt.scatter(X, Y)  # (x, y) 점을 그립니다.
plt.plot([0, 10], [beta_1, 10 * beta_0 + beta_1], c='r')  # y = beta_0 * x + beta_1 에 해당하는 선을 그립니다.

plt.xlim(0, 10)  # 그래프의 X축을 설정합니다.
plt.ylim(0, 10)  # 그래프의 Y축을 설정합니다.


#@@ scikit-learn을 이용한 linear regression (LR)
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
def loss(x, y, beta_0, beta_1):
    N = len(x)
    x = np.array(x)
    y = np.array(y)
    y_predict = beta_0 * x + beta_1
    total_loss = np.sum((y - y_predict) ** 2)
    return total_loss


X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513,
     5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441,
     5.19692852]

train_X = np.array(X).reshape(-1, 1)
train_Y = np.array(Y)

'''
여기에서 모델을 트레이닝합니다.
'''
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)

'''
loss가 최소가 되는 직선의 기울기와 절편을 계산함
'''
beta_0 = lrmodel.coef_[0]  # lrmodel로 구한 직선의 기울기
beta_1 = lrmodel.intercept_  # lrmodel로 구한 직선의 y절편

print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("Loss: %f" % loss(X, Y, beta_0, beta_1))

plt.scatter(X, Y)  # (x, y) 점을 그립니다.
plt.plot([0, 10], [beta_1, 10 * beta_0 + beta_1], c='r')  # y = beta_0 * x + beta_1 에 해당하는 선을 그립니다.

plt.xlim(0, 10)  # 그래프의 X축을 설정합니다.
plt.ylim(0, 10)  # 그래프의 Y축을 설정합니다


#@@ multi-linear regression(MLR)
"""
여러개의 변수(input variable)와 알고자하는 y(output : 해답/응답)와의 관계를 파악하는 것
input = vector, output = scalar
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

'''
./data/Advertising.csv 에서 데이터를 읽어, X와 Y를 만듭니다.

X는 (200, 3) 의 shape을 가진 2차원 np.array,
Y는 (200,) 의 shape을 가진 1차원 np.array여야 합니다.

X는 FB, TV, Newspaper column 에 해당하는 데이터를 저장해야 합니다.
Y는 Sales column 에 해당하는 데이터를 저장해야 합니다.
'''

import csv
csvreader = csv.reader(open("data/Advertising.csv")) #예시파일이라 해당 코드에서는 안불러짐

x = []
y = []

next(csvreader)
for line in csvreader :
    x_i = [ float(line[1]), float(line[2]), float(line[3]) ]
    y_i = float(line[4])
    x.append(x_i)
    y.append(y_i)

X = np.array(x)
Y = np.array(y)

lrmodel = LinearRegression()
lrmodel.fit(X, Y)

beta_0 = lrmodel.coef_[0] # 0번째 변수에 대한 계수 (페이스북)
beta_1 = lrmodel.coef_[1] # 1번째 변수에 대한 계수 (TV)
beta_2 = lrmodel.coef_[2] # 2번째 변수에 대한 계수 (신문)
beta_3 = lrmodel.intercept_ # y절편 (기본 판매량)

print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("beta_2: %f" % beta_2)
print("beta_3: %f" % beta_3)

def expected_sales(fb, tv, newspaper, beta_0, beta_1, beta_2, beta_3):
    '''
    FB에 fb만큼, TV에 tv만큼, Newspaper에 newspaper 만큼의 광고비를 사용했고,
    트레이닝된 모델의 weight 들이 beta_0, beta_1, beta_2, beta_3 일 때
    예상되는 Sales 의 양을 출력합니다.
    '''
    y = beta_0*fb + beta_1*tv + beta_2*newspaper + beta_3
    return y

print("예상 판매량: %f" % expected_sales(10, 12, 3, beta_0, beta_1, beta_2, beta_3))

#@@ 다항식 회귀분석(polynominal linear regression)
"""
선형식을 곡선으로 표현하고 싶었음. 그러다보니 다항식으로 회귀분석을 하고자 한 것.
MSE : 평균제곱오차; 통계적 추정에 대한 정확성의 지표로 사용됨.
교차검증 : 모델이 결과를 잘 예측하는 지 알아보기 위해 전체 데이터를 training & test set
으로 나누어 모델에 넣고 성능을 평가하는 방법. 트레이닝 데이터는 모델을 학습시킬 때 사용되고, 테스트
데이터는 학습된 모델을 검증할 때 사용됨.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
'''
./data/Advertising.csv 에서 데이터를 읽어, X와 Y를 만듭니다.

X는 (200, 3) 의 shape을 가진 2차원 np.array,
Y는 (200,) 의 shape을 가진 1차원 np.array여야 합니다.

X는 FB, TV, Newspaper column 에 해당하는 데이터를 저장해야 합니다.
Y는 Sales column 에 해당하는 데이터를 저장해야 합니다.
'''
import csv
csvreader = csv.reader(open("data/Advertising.csv")) #예시파일이라 해당 코드에서는 안불러짐

x = []
y = []

next(csvreader)
for line in csvreader :
    x_i = [ float(line[1]), float(line[2]), float(line[3]) ]
    y_i = float(line[4])
    x.append(x_i)
    y.append(y_i)

X = np.array(x)
Y = np.array(y)


# 다항식 회귀분석을 진행하기 위해 변수들을 조합합니다.
X_poly = []
for x_i in X:
    X_poly.append([
        x_i[0] ** 2, # X_1^2
        x_i[1] ** 2,
        x_i[2] ** 2,
        x_i[1], # X_2
        x_i[0],
        x_i[2],
        x_i[1] * x_i[0],
        x_i[1] * x_i[2], # X_2 * X_3
        x_i[0] * x_i[2]
    ])

# X, Y를 80:20으로 나눕니다. 80%는 트레이닝 데이터, 20%는 테스트 데이터입니다.
x_train, x_test, y_train, y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=0)

# x_train, y_train에 대해 다항식 회귀분석을 진행합니다.
lrmodel = LinearRegression()
lrmodel.fit(x_train, y_train)

#x_train에 대해, 만든 회귀모델의 예측값을 구하고, 이 값과 y_train 의 차이를 이용해 MSE를 구합니다.
predicted_y_train = lrmodel.predict(x_train)
mse_train = mean_squared_error(y_train, predicted_y_train)
print("MSE on train data: {}".format(mse_train))

# x_test에 대해, 만든 회귀모델의 예측값을 구하고, 이 값과 y_test 의 차이를 이용해 MSE를 구합니다. 이 값이 1 미만이 되도록 모델을 구성해 봅니다.
predicted_y_test = lrmodel.predict(x_test)
mse_test = mean_squared_error(y_test, predicted_y_test)
print("MSE on test data: {}".format(mse_test))


#@@ 실습 : 영어단어 코퍼스 분석하기
import operator
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import elice_utils
import csv


def main():
    words = read_data()
    words = sorted(words, key=lambda x: x[1], reverse=True)  # words.txt 단어를 빈도수 순으로 정렬합니다.

    # 정수로 표현된 단어를 X축 리스트에, 각 단어의 빈도수를 Y축 리스트에 저장합니다.
    X = list(range(1, len(words) + 1))
    Y = [x[1] for x in words]

    # X, Y 리스트를 array로 변환한 후 각 원소 값에 log()를 적용합니다.
    X, Y = np.array(X), np.array(Y)

    X, Y = np.log(X), np.log(Y)

    print(X)
    print(Y)

    # 기울기와 절편을 구한 후 그래프와 차트를 출력합니다.
    slope, intercept = do_linear_regression(X, Y)
    draw_chart(X, Y, slope, intercept)

    return slope, intercept


# read_data() - words.txt에 저장된 단어와 해당 단어의 빈도수를 리스트형으로 변환합니다.
def read_data():
    # words.txt 에서 단어들를 읽어,
    # [[단어1, 빈도수], [단어2, 빈도수] ... ]형으로 변환해 리턴합니다.
    words = []

    filename = 'words.txt'

    x = []
    y = []

    with open(filename) as data:
        lines = data.readlines()

    for line in lines:
        word = line.replace('\n', '').split(',')
        word[1] = int(word[1])
        words.append(word)
    return words


# do_linear_regression() - 임포트한 sklearn 패키지의 함수를 이용해 그래프의 기울기와 절편을 구합니다.
def do_linear_regression(X, Y):
    # do_linear_regression() 함수를 작성하세요.
    X = X.reshape(-1, 1)
    li = LinearRegression()
    li.fit(X, Y)

    slope = li.coef_[0]  # 2번째 변수에 대한 계수 (신문)
    intercept = li.intercept_  #

    return (slope, intercept)


# draw_chart() - matplotlib을 이용해 차트를 설정합니다.
def draw_chart(X, Y, slope, intercept):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X, Y)

    # 차트의 X, Y축 범위와 그래프를 설정합니다.
    min_X = min(X)
    max_X = max(X)
    min_Y = min_X * slope + intercept
    max_Y = max_X * slope + intercept
    plt.plot([min_X, max_X], [min_Y, max_Y],
             color='red',
             linestyle='--',
             linewidth=3.0)

    # 기울과와 절편을 이용해 그래프를 차트에 입력합니다.
    ax.text(min_X, min_Y + 0.1, r'$y = %.2lfx + %.2lf$' % (slope, intercept), fontsize=15)

    plt.savefig('chart.png')
    elice_utils.send_image('chart.png')


if __name__ == "__main__":
    main()


#@@ 확률로 파이 계산하기 - 몬테칼로 방법 이용
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.figure(figsize=(5, 5)) #figure 크기 설정

    X = []
    Y = []

    # N을 10배씩 증가할 때 파이 값이 어떻게 변경되는지 확인해보세요.
    # N이 증가할수록 실제 파이값에 수렴함
    N = 1000

    for i in range(N): #1000개의 난수 구하기
        X.append(np.random.rand() * 2 - 1) # np.random.rand() * 2 - 1 : [0,1] -> [0,2]-> [-1,1]
        Y.append(np.random.rand() * 2 - 1)
    X = np.array(X)
    Y = np.array(Y)
    distance_from_zero = np.sqrt(X * X + Y * Y)
    is_inside_circle = distance_from_zero <= 1 #bull로 나오게됨.

    print("Estimated pi = %f" % (np.average(is_inside_circle) * 4))

    plt.scatter(X, Y, c=is_inside_circle)
    plt.savefig('circle.png')
    elice_utils.send_image('circle.png')


if __name__ == "__main__":
    main()

#@@유방암 계산하기
'''
>>> 0.8
>>> 0.004
>>> 0.1
3.11%

sensitivity - 검사의 민감성을 뜻합니다. 유방암 보유자를 대상으로 검사 결과가 양성으로 표시될 확률입니다. 0부터 1 사이의 값을 갖습니다
prior_prob - 총 인구를 기준으로 유방암을 가지고 있을 사전 확률(prior probability)입니다. 0.004 정도로 매우 낮은 값입니다.
false_alarm - 실제로는 암을 갖고 있지 않지만 유방암이라고 진단될 확률입니다. 0.1 정도로 생각보다 높은 값입니다.

'''
def main():
    sensitivity = float(input())
    prior_prob = float(input())
    false_alarm = float(input())

    print("%.2lf%%" % (100 * mammogram_test(sensitivity, prior_prob, false_alarm)))

def mammogram_test(sensitivity, prior_prob, false_alarm):
    p_a1_b1 = sensitivity # p(A = 1 | B = 1)

    p_b1 = prior_prob    # p(B = 1)

    p_b0 = 1 - prior_prob    # p(B = 0)

    p_a1_b0 = false_alarm # p(A = 1|B = 0)

    p_a1 = p_a1_b0*p_b0 + p_a1_b1*p_b1   # p(A = 1)

    p_b1_a1 = p_a1_b1*p_b1/p_a1 # p(B = 1|A = 1)

    return p_b1_a1

if __name__ == "__main__":
    main()

#@@ 나이브 베이즈 분류기
import re
import math
import numpy as np


def main():
    M1 = {'r': 0.7, 'g': 0.2, 'b': 0.1}  # M1 기계의 사탕 비율
    M2 = {'r': 0.3, 'g': 0.4, 'b': 0.3}  # M2 기계의 사탕 비율

    test = {'r': 4, 'g': 3, 'b': 3}

    print(naive_bayes(M1, M2, test, 0.7, 0.3))


def naive_bayes(M1, M2, test, M1_prior, M2_prior):
    M1_likelihood = M1['r']**test['r']*M1['g']**test['g']*M1['b']**test['b']
    M1_posterior = M1_likelihood*M1_prior
    M2_likelihood = M2['r'] ** test['r'] * M2['g'] ** test['g'] * M2['b'] ** test['b']
    M2_posterior = M2_likelihood * M2_prior
    M1_normalized  = M1_posterior / (M1_posterior+M2_posterior)
    M2_normalized = M2_posterior / (M1_posterior + M2_posterior)
    return [M1_normalized, M2_normalized]

"""
for loop을 통해 일반화 하는 방법은 다음 코드에 작성하기로 함.
"""


if __name__ == "__main__":
    main()


#@@Bag of words 실습
import re

special_chars_remover = re.compile("[^\w'|_]")


def main():
    sentence = input() # "Bag-of-Words 모델을 Python으로 직접 구현하겠습니다."
    bow = create_BOW(sentence)

    print(bow)


def create_BOW(sentence):
    bow = {}
    sentence_lowered = sentence.lower()
    sentence_without_special = remove_special_characters(sentence_lowered)
    splitted_sentence = sentence_without_special.split()
    splitted_sentence_filtered = [
        token
        for token in splitted_sentence
        if len(token) >= 1
    ]
    for token in splitted_sentence_filtered:
        bow.setdefault(token, 0) #token이 없으면 0으로 reset하고,
        bow[token] += 1 #위 코드 실행 후 1을 더해라. (token이 있으면 443번 코드는 무시됨)
'''
    for token in splitted_sentence_filtered:
        if token not in bow:
            bow[token] = 1
        else:
            bow[token] += 1
'''
    return bow

def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)


if __name__ == "__main__":
    main()


#@@ sentiment classifier



#@@ PCA 실습
import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy as np

def main():
    X, attributes = input_data()
    pca_array = normalize(X)
    pca, pca_array = run_PCA(X, 2)
    visualize_2d_wine(pca_array)


def input_data():
    X = []
    attributes = []
    csvreader = csv.reader(open("data/wine.csv"))
    for line in csvreader:
        float_numbers = [float(x) for x in line]
        X.append(float_numbers)

    return np.array(X), attributes


def run_PCA(X, num_components):
    pca = sklearn.decomposition.PCA(n_components = num_components)
    pca.fit(X)
    pca_array = pca.transform(X)

    return pca, pca_array


def normalize(X):
    '''
    각각의 feature에 대해,
    178개의 데이터에 나타나는 해당하는 feature의 값이 최소 0, 최대 1이 되도록
    선형적으로 데이터를 이동시킵니다.
    '''
    for i in range(X.shape[1]):
        X[:,i] = X[:, i] - np.min(X[:,i])
        X[:,i]= X[:, i]/np.max(X[:,i])
    return X


def visualize_2d_wine(X):
    '''X를 시각화하는 코드를 구현합니다.'''
    plt.scatter(X[:,0], X[:,1])
    plt.show()

if __name__ == '__main__':
    main()

#@@K-means 클러스터링 실습
def kmeans(X, num_clusters, initial_centroid_indices):
    import time
    N = len(X)
    centroids = X[initial_centroid_indices]
    labels = np.zeros(N) #초기값은 다 0으로 셋팅

    while True:
        '''
        Step 1. 각 데이터 포인트 i 에 대해 가장 가까운
        중심점을 찾고, 그 중심점에 해당하는 클러스터를 할당하여
        labels[i]에 넣습니다.
        가까운 중심점을 찾을 때는, 유클리드 거리를 사용합니다.
        미리 정의된 distance 함수를 사용합니다.
        '''
        #step1
        is_changed = False
        for i in range(N):
            distances =[]
            for k in range(num_clusters):
            #X['i]와 centroids[k](k번째 중심점)와의 거리
                k_dist = distance(X[i], centroids[k])
                distances.append(k_dist)
            if labels[i] != np.argmin(distances):
                is_changed = True
            labels[i] = np.argmin(distances) #distances의 최솟값
        '''
        Step 2. 할당된 클러스터를 기반으로 새로운 중심점을 계산합니다.
        중심점은 클러스터 내 데이터 포인트들의 위치의 *산술 평균*
        으로 합니다.
        '''
        for k in range(num_clusters):
            x = X[labels == k][:,0] #X의 label이 k인 값을 가진 index의 X[index]를 추출
            y = X[labels == k][:,1]

            x = np.mean(x)
            y = np.mean(y)
            centroids[k] = [x,y]
        '''
        Step 3. 만약 클러스터의 할당이 바뀌지 않았다면 알고리즘을 끝냅니다.
        아니라면 다시 반복합니다.
        '''
        if not is_changed:
            break
    return labels
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def normalize(X):
    for dim in range(len(X[0])):
        X[:, dim] -= np.min(X[:, dim])
        X[:, dim] /= np.max(X[:, dim])
    return X


'''
이전에 더해, 각각의 데이터 포인트에 색을 입히는 과정도 진행합니다.
'''


def visualize_2d_wine(X, labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:,0], X[:,1], c=labels)

if __name__ == '__main__':
    main()

#@@ Naive Bayes Classifer
import numpy as np


# 리스트 안에 값들을 정규화 합니다.
def normalization(x):
    return [element / sum(x) for element in x]


# 1. P(“스팸 메일”) 의 확률을 구하세요.
p_spam = 8/20

# 2. P(“확인” | “스팸 메일”) 의 확률을 구하세요.
p_confirm_spam = 5/8

# 3. P(“정상 메일”) 의 확률을 구하세요.
p_ham = 12/20

# 4. P(“확인” | "정상 메일" ) 의 확률을 구하세요.
p_confirm_ham = 2/12

# 5. P( "스팸 메일" | "확인" ) 의 확률을 구하세요.
p_spam_confirm = 2/7

# 6. P( "정상 메일" | "확인" ) 의 확률을 구하세요.
p_ham_confirm = 5/7

print("P(spam|confirm) = ",p_spam_confirm, "\nP(ham|confirm) = ",p_ham_confirm, "\n")

# 두 값을 비교하여 확인 키워드가 스팸에 가까운지 정상 메일에 가까운지 확인합니다.
value = [p_spam_confirm, p_ham_confirm]
result = normalization(value)

print("P(spam|confirm) normalization = ",result[0], "\nP(ham|confirm) normalization = ",result[1], "\n")

if p_spam_confirm > p_ham_confirm:
    print( round(result[0] * 100, 2), "% 의 확률로 스팸 메일에 가깝습니다.")
else :
    print( round(result[1] * 100, 2), "% 의 확률로 일반 메일에 가깝습니다.")


#@@ 선형회귀 직접 구현하기
import numpy as np
import elice_utils
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")
eu = elice_utils.EliceUtils()
learning_rate = 1e-4
iteration = 10000

x = np.array(
    [[8.70153760], [3.90825773], [1.89362433], [3.28730045], [7.39333004], [2.98984649], [2.25757240], [9.84450732],
     [9.94589513], [5.48321616]])
y = np.array(
    [[5.64413093], [3.75876583], [3.87233310], [4.40990425], [6.43845020], [4.02827829], [2.26105955], [7.15768995],
     [6.29097441], [5.19692852]])


##입력값(x)과 변수(a,b)를 바탕으로 예측값을 출력하는 함수를 만들어 봅니다.
def prediction(a, b, x):
    # 1.Numpy 배열 a,b,x를 받아서 'x*(transposed)a + b'를 계산하는 식을 만듭니다.
    equation = x * a.T + b
    return equation


##변수(a,b)의 값을 어느정도 업데이트할 지를 정해주는 함수를 만들어 봅니다.
def update_ab(a, b, x, error, lr):
    ## a를 업데이트하는 규칙을 만듭니다..
    delta_a = -(lr * (2 / len(error)) * (np.dot(x.T, error)))
    ## b를 업데이트하는 규칙을 만듭니다.
    delta_b = -(lr * (2 / len(error)) * np.sum(error))

    return delta_a, delta_b


# 반복횟수만큼 오차(error)를 계산하고 a,b의 값을 변경하는 함수를 만들어 봅니다.
def gradient_descent(x, y, iters):
    ## 초기값 a= 0, b=0
    a = np.zeros((1, 1))
    b = np.zeros((1, 1))

    for i in range(iters):
        # 2.실제 값 y와 prediction 함수를 통해 예측한 예측값의 차이를 error로 정의합니다.
        error = y - prediction(a, b, x)
        # 3.위에서 정의한 함수를 이용하여 a와 b 값의 변화값을 저장합니다.
        a_delta, b_delta = update_ab(a, b, x, error, lr=learning_rate)
        ##a와 b의 값을 변화시킵니다.
        a -= a_delta
        b -= b_delta

    return a, b


##그래프를 시각화하는 함수입니다.
def plotting_graph(x, y, a, b):
    y_pred = a[0, 0] * x + b
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.savefig("test.png")
    eu.send_image("test.png")


##실제 진행 절차를 확인할 수 있는 main함수 입니다.
def main():
    a, b = gradient_descent(x, y, iters=iteration)
    print("a:", a, "b:", b)
    plotting_graph(x, y, a, b)
    return a, b


main()