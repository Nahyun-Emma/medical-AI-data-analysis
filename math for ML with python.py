import numpy as np

# @@ numpy vector 만들기
def main():
    print(tutorial_1st())
    print(tutorial_2nd())

def tutorial_1st():
    """
    지시사항 1.
    tutorial_1st() 함수 안에 5번째 값만 1로 가지고 이외의 값은 0을 가지는
    길이 10의 벡터를 선언하세요.
    """
    A = np.zeros(10)
    A[4] = 1
    return A


def tutorial_2nd():
    """
    지시사항 2.
    tutorial_2nd() 함수 안에 10~49의 range를 가지는 벡터를 선언하세요.
    """

    B = np.arange(10, 50)
    return B

if __name__ == "__main__":
    main()

#@@ pandas활용
def main():
    print(pandas_tutorial())

def pandas_tutorial():
    '''
    지시사항: `[2,4,6,8,10]`의 리스트를 pandas의 Series 자료구조로 선언하세요.
    '''
    A = pd.Series([2, 4, 6, 8, 10])
    return A

if __name__ == "__main__":
    main()


#@@matplotlib
def main():
    matplotlib_tutorial()


def matplotlib_tutorial():
    '''
    지시사항: data [1,2,3,4]를 정의하여서, plt.plot을 활용해 데이터를 시각화해보세요.
    '''
    data = [1, 2, 3, 4]
    plt.plot(data)

    # 엘리스에서 이미지를 출력하기 위해서 필요한 코드입니다.
    plt.savefig("plot.png")
    elice_utils.send_image("plot.png")


if __name__ == "__main__":
    main()

#@@numpy vector 연산
def main():
    data = [1, 2, 3]

    print(calculate_norm(data))
    print(unit_vector(data))


def calculate_norm(data):
    '''
    지시사항1: 함수 안에 주어진 데이터 data의 크기(scale) 값을 출력하는 함수를 채우시오.
    '''
    return np.linalg.norm(data)


def unit_vector(data):
    '''
    지시사항2: 함수 입력으로 주어진 data 벡터를 단위 벡터(unit vector)로 바꾸어 출력하세요.
    '''

    return np.array(data) / np.linalg.norm(data)


if __name__ == "__main__":
    main()

#@@ 행렬 연산
def main():
    print(normalize())


def normalize():
    '''
    지시사항: 함수 안에서 먼저 (5,5)의 랜덤 행렬을 선언하세요.그리고 선언한 행렬의 최대값과 최소값을 기억하고, 이것을 통해 선언했던 랜덤 행렬을 normalize하세요.
    '''

    Z = np.random.random((5, 5))
    Zmax, Zmin = Z.max(), Z.min()
    Z = (Z - Zmin) / (Zmax - Zmin)

    return Z


if __name__ == "__main__":
    main()

#@@ 행렬 곱 함수 만들기
def main():
    print(multiply())


def multiply():
    '''
    함수 안에 먼저 1으로 채워진 행렬 2개를 선언합니다. 하나는 (5,3)의 크기, 다른 하나는 (3,2)로 합니다. 그리고 앞서 선언한 두 행렬의 곱을 합니다.
    '''
    A = np.ones((5, 3))
    B = np.ones((3, 2))
    Z = np.dot(A, B)

    return Z


if __name__ == "__main__":
    main()

#@@함수의 미분
def main():
    x = 5
    print(derivative(x))


def derivative(x):
    '''
    먼저 다항함수 $x^2+1$을 선언하세요.그리고 선언한 다항함수의 미분 함수를 정의하고, 이것에  x를 넣은 값을 출력하세요.
    '''
    p = np.poly1d([1,0,1])
    print(p)

    q = p.deriv()
    print(q)

    return q(x)




if __name__ == "__main__":
    main()

#@@다항함수의 미분 근사
def main():
    x = 5
    print(d_fun(x))

def d_fun(x):
    '''
    함수 안에 x^2+1의 다항함수를 선언합니다. 그리고 함수의 극한에 활용할 분모값 h=1e-5를 활용해 미분값을 근사합니다. 미분값의 근사는 미분계수를 구할 때 활용한 극한식을 참고하세요.
    '''
    p = np.poly1d([1,0,1])
    print(p)
    h = 1e-5

    return (p(x+h)-p(x))/h


if __name__ == "__main__":
    main()

#@@편미분
def main():
    x = 5
    y = 2
    print(d_fun(x,y, respect = 'x'))
    print(d_fun(x,y, respect = 'y'))

def fun(x,y):

    return x**3 * y

def d_fun(x,y, respect = 'y'):
    '''
     x,y의 점에서 y에 대한 편미분 값을 구하는 함수입니다. 함수의 극한을 응용합니다. 주어진 h값을 활용해, x에 대한 미분은 respect='x'일때, 그리고 y에 대한 미분은 respect='y'일 때입니다.
    '''
    h = 1e-5

    if respect == 'x':
        answer = (fun(x+h,y)-fun(x,y))/h
    elif respect == 'y':
        answer = (fun(x,y+h)-fun(x,y))/h
    else:
        raise NotImplementedError

    return answer



if __name__ == "__main__":
    main()