import numpy as np

def vpfcwbc(X, n_clusters=4, m=2.0, lambda_bias=0.5, max_iter=50):
    N, D = X.shape
    U = np.random.rand(N, n_clusters)
    U = U / np.sum(U, axis=1, keepdims=True)
    C = np.random.randn(n_clusters, D)
    b = np.zeros(N)
    for it in range(max_iter):
        for j in range(n_clusters):
            num = np.sum((U[:, j]**m).reshape(-1,1)*(X - b.reshape(-1,1)), axis=0)
            den = np.sum(U[:, j]**m)
            C[j] = num / (den + 1e-8)
        dist = np.zeros((N, n_clusters))
        for j in range(n_clusters):
            dist[:, j] = np.linalg.norm(X - C[j], axis=1)**2 + lambda_bias*b**2
        U = 1.0 / (dist + 1e-8)
        U = U / np.sum(U, axis=1, keepdims=True)
        b = np.mean(X - U.dot(C), axis=1)
    labels = np.argmax(U, axis=1)
    return labels.reshape(-1), b
