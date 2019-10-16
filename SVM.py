import numpy as np
from scipy.signal import convolve2d
from skimage.transform import resize
from math import sqrt
from sklearn import svm


def extract_hog(image):
    img = resize(image, (64, 64), anti_aliasing=True, mode='reflect')
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y = 0.299 * r + 0.587 * g + 0.114 * b

    x_ker = np.array([[-1, 0, 1]])
    y_ker = x_ker.transpose()

    grad_x = convolve2d(Y, x_ker, boundary='symm', mode='same')
    grad_y = convolve2d(Y, y_ker, boundary='symm', mode='same')
    grad = np.hypot(grad_x, grad_y)

    angle = np.arctan2(grad_y, grad_x)

    angle[angle < 0] += np.pi

    cellRows = 8
    cellCols = 8
    binCount = 9

    histograms = np.zeros((angle.shape[0] // cellRows,
                           angle.shape[1] // cellCols,
                           binCount))
    for i in range(angle.shape[0] // cellRows):
        for j in range(angle.shape[1] // cellCols):
            histograms[i, j, :], _ = np.histogram(angle[i*cellRows:(i+1)*cellRows, j*cellCols:(j+1)*cellCols],
                                                  bins=binCount,
                                                  range=(0, np.pi),
                                                  weights=grad[i*cellRows:(i+1)*cellRows, j*cellCols:(j+1)*cellCols])

    blockRowCells = 2
    blockColCells = 2
    eps = 0.001

    normalized = np.zeros((histograms.shape[0] - blockRowCells,
                           histograms.shape[1] - blockColCells,
                           blockRowCells * blockColCells * histograms.shape[2]))

    for i in range(histograms.shape[0] - blockRowCells):
        for j in range(histograms.shape[1] - blockColCells):
            v = histograms[i:i+blockRowCells, j:j+blockColCells, :].flatten()
            v = (1 / sqrt(np.linalg.norm(v) ** 2 + eps)) * v
            normalized[i, j, :] = v

    return normalized.flatten()


def fit_and_classify(train_features, train_labels, test_features):
    clf = svm.SVC(C=10, tol=1e-3, gamma='scale')
    #  clf = svm.LinearSVC(C=1.0, tol=1e-4)
    clf.fit(train_features, train_labels)
    y = clf.predict(test_features)
    return y
