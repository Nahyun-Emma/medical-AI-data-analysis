import numpy as np

#@@ matrix 만들고 Normalize하기
def main():
    print(matrix_tutorial())


def matrix_tutorial():
    A = np.array([[1, 4, 5, 8], [2, 1, 7, 3], [5, 4, 5, 9]])

    # normalize : 원소의 합이 1이 되도록 적용하는 것
    A = A / np.sum(A)
    # variance :  np.var()
    return np.var(A)


if __name__ == "__main__":
    main()

#@@ 전치행렬, 역행렬 만들기
def main():
    A = get_matrix()
    print(matrix_tutorial(A))


def get_matrix():
    mat = []
    [n, m] = [int(x) for x in input().strip().split(" ")]  # n :행, m : 열
    for i in range(n):
        row = [int(x) for x in input().strip().split(" ")]
        mat.append(row)
    return np.array(mat)


def matrix_tutorial(A):
    # 아래 코드를 완성하세요.
    B = np.transpose(A)  # or A.T
    try:
        C = np.linalg.inv(B)f
    except:
        return "not invertible"
    return np.sum(C > 0)  # C>0 을 함으로써 행렬의 모든 원소의 bull 가능. np에서는 True =1, False 0으로 인식하므로


if __name__ == "__main__":
    main()


#@@벡터 연산으로 그림 그리기
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def circle(P):
    return np.linalg.norm(P) - 1  # 밑의 코드와 동일하게 동작합니다.
    # return np.sqrt(np.sum(P * P)) - 1


def diamond(P):
    return np.abs(P[0]) + np.abs(P[1]) - 1


def smile(P):
    def left_eye(P):
        eye_pos = P - np.array([-0.5, 0.5])
        return np.sqrt(np.sum(eye_pos * eye_pos)) - 0.1

    def right_eye(P):
        eye_pos = P - np.array([0.5, 0.5])
        return np.sqrt(np.sum(eye_pos * eye_pos)) - 0.1

    def mouth(P):
        if P[1] < 0:
            return np.sqrt(np.sum(P * P)) - 0.7
        else:
            return 1

    return circle(P) * left_eye(P) * right_eye(P) * mouth(P)


def checker(P, shape, tolerance):
    return abs(shape(P)) < tolerance


def sample(num_points, xrange, yrange, shape, tolerance):
    accepted_points = []
    rejected_points = []

    for i in range(num_points):
        x = np.random.random() * (xrange[1] - xrange[0]) + xrange[0]
        y = np.random.random() * (yrange[1] - yrange[0]) + yrange[0]
        P = np.array([x, y])

        if (checker(P, shape, tolerance)):
            accepted_points.append(P)
        else:
            rejected_points.append(P)

    return np.array(accepted_points), np.array(rejected_points)


xrange = [-1.5, 1.5]  # X축 범위입니다.
yrange = [-1.5, 1.5]  # Y축 범위입니다.
accepted_points, rejected_points = sample(
    100000,  # 점의 개수를 줄이거나 늘려서 실행해 보세요. 너무 많이 늘리면 시간이 오래 걸리는 것에 주의합니다.
    xrange,
    yrange,
    smile,  # smile을 circle 이나 diamond 로 바꿔서 실행해 보세요.
    0.005)  # Threshold를 0.01이나 0.0001 같은 다른 값으로 변경해 보세요.

plt.figure(figsize=(xrange[1] - xrange[0], yrange[1] - yrange[0]),
           dpi=150)  # 그림이 제대로 로드되지 않는다면 DPI를 줄여보세요.

plt.scatter(rejected_points[:, 0], rejected_points[:, 1], c='lightgray', s=0.1)
plt.scatter(accepted_points[:, 0], accepted_points[:, 1], c='black', s=1)
