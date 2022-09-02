import numpy as np
import modern_robotics as mr
import scipy.linalg as sl
import matplotlib.pyplot as plt

######################### HELPER FUNCTIONS ###############################################


def calc_sum(r, b, X, k):
    n = len(r)
    result = np.zeros((n+4, n+4))
    for i in range(n):
        result += k[i]*np.outer(r[i] - np.dot(X, b[i]), r[i])
    return result


def build_r(n):
    r = np.zeros((n, n+4))
    for lmr in range(n):
        r[lmr] = np.concatenate(
            (np.zeros(3, dtype=float), np.ones(1, dtype=float), -np.eye(n)[lmr]))
    return r


def adjoint(X, U):
    return np.dot(X, np.dot(U, np.linalg.inv(X)))


def anti_symmetric(M):
    return (M-M.T)/2


def P(M):
    n = len(M) - 4
    A = np.zeros((n+4, n+4))
    A1 = M[0:3, 0:3]
    A2 = M[0:3, 3:n+4]
    A[0:3, 0:3] = anti_symmetric(A1)
    A[0:3, 3:n+4] = A2
    return A


# initializing input vectors
landmarks = np.random.uniform(-10, 10, size=(3, 16))
w = np.array([0, 0, 1])
v = np.array([0, 1, 0])
n = len(landmarks.T)
k = np.ones(n)*5/22
bw = np.array([-0.02, 0.02, 0.01])
bv = np.array([0.2, -0.1, 0.1])
kw = 0.02
kv = 1
S = np.concatenate((w, v))
Su = np.concatenate((bw, bv))
bu = np.zeros((n+4, n+4))
bu[0:4, 0:4] = mr.VecTose3(Su)
U = np.zeros((n+4, n+4))
U[0:4, 0:4] = mr.VecTose3(S)
K = np.diag(np.concatenate(([kw, kw, kw, kv], np.zeros(n))))
Uy = U + bu
time = np.linspace(0, 100, 10001)

# Initializing state and r
X_init = np.eye(n+4)
X_init[2, 3] = 10
X_init[0:3, 4:n+4] = landmarks
r = build_r(n)

# generating real trajectory
X = []

for t in time:
    new_X = np.dot(X_init, sl.expm(U*t))
    X.append(new_X)

X = np.array(X)

# Initializing estimated state
S_hat = np.array([0, 0, 1, 0, 0, 0])*0.2*np.pi
X_hat_init = np.eye(n+4)
X_hat_init[0:4, 0:4] = mr.MatrixExp6(mr.VecTose3(S_hat))
X_hat_init[0:3, 4:n+4] = np.zeros((3, n))


# generating estimated trajectory
X_hat = []
bu_hat = []

for i, t in enumerate(time):
    if i == 0:
        X_hat.append(X_hat_init)
        bu_hat.append(np.zeros((n+4, n+4)))
    else:
        b = np.dot(np.linalg.inv(X[i-1]), r.T).T
        A = calc_sum(r, b, X_hat[i-1], k)
        delta = -adjoint(np.linalg.inv(X_hat[i-1]), P(A))
        X_hat.append(
            np.dot(X_hat[i-1], sl.expm((Uy - delta - bu_hat[i-1])*time[1])))
        bu_hat.append(
            (bu_hat[i-1]-time[1]*np.dot(P(adjoint(X_hat[i-1].T, A)), K)))

X_hat = np.array(X_hat)
bu_hat = np.array(bu_hat)


# plotting results

X_tild = np.array([np.dot(X[i], np.linalg.inv(X_hat[i]))
                  for i, _ in enumerate(X)])
R_tild = X_tild[:, 0:3, 0:3]
p_tild = X_tild[:, 0:3, 3]
pi_tild = X_tild[:, 0:3, 4:n+4]
bu_tild = bu_hat - bu

norm_R = np.array([np.linalg.norm(R_tild[i] - np.eye(3))
                  for i, _ in enumerate(R_tild)])

norm_p = np.array([np.linalg.norm(p_tild[i])
                   for i, _ in enumerate(p_tild)])
norm_pi = np.array([np.linalg.norm(np.dot(R_tild[i].T, pi_tild[i] - p_tild[i].reshape(3, 1)).T, axis=1)
                    for i, _ in enumerate(X)])

b_norm = np.array([np.linalg.norm([bu_hat[i] - bu])
                  for i, _ in enumerate(X)])
bw_norm = np.array([np.linalg.norm([bu_hat[i, 0:3, 0:3] - bu[0:3, 0:3]])
                    for i, _ in enumerate(X)])
bv_norm = np.array([np.linalg.norm([bu_hat[i, 0:3, 3] - bu[0:3, 3]])
                    for i, _ in enumerate(X)])

ei = np.array([np.dot(R_tild[i].T, pi_tild[i] - p_tild[i].reshape(3, 1)).T
               for i, _ in enumerate(X)])

ei_sum = np.zeros(len(ei))

for i in range(len(ei)):
    ei_sum[i] = np.linalg.norm(np.sum(ei[i]*5/22, axis=0))

# np.savetxt("time.csv", time, delimiter=",")
# np.savetxt("bv.csv", bv_norm, delimiter=",")
# np.savetxt("bw.csv", bw_norm, delimiter=",")
# np.savetxt("R_tild.csv", norm_R, delimiter=",")
# np.savetxt("p_tild.csv", norm_p, delimiter=",")
# np.savetxt("ei.csv", norm_pi, delimiter=",")
# np.savetxt("ei_sum.csv", ei_sum, delimiter=",")

plt.figure(dpi=100)
plt.subplot(2, 1, 1)
plt.ylabel(r"$\Vert\mathit{ I - \tilde R}\Vert_{F}$", fontsize=18)
plt.grid()
plt.plot(time, norm_R, "blue")

plt.subplot(2, 1, 2)
plt.xlabel(r"$t(s)$", fontsize=18)
plt.ylabel(r"$\Vert\mathit{\tilde p}\Vert$", fontsize=18)
plt.plot(time, norm_p, "blue")
plt.grid()


plt.figure(dpi=100)
for p in norm_pi.T:
    plt.xlabel(r"$t(s)$", fontsize=18)
    plt.ylabel(r"$\Vert\mathit{\tilde \eta_{i}}\Vert$", fontsize=18)
    plt.plot(time, p)
plt.grid()

plt.figure(dpi=100)
plt.subplot(2, 1, 1)
plt.ylabel(r"$\Vert\mathit{\tilde b_{\Omega}}\Vert_{F}$", fontsize=18)
plt.grid()
plt.plot(time, bw_norm, "blue")

plt.subplot(2, 1, 2)
plt.xlabel(r"$t(s)$", fontsize=18)
plt.ylabel(r"$\Vert\mathit{\tilde b_{v}}\Vert$", fontsize=18)
plt.plot(time, bv_norm, "blue")
plt.grid()
plt.show()

plt.figure(dpi=100)
plt.plot(time, ei_sum)
plt.grid()
plt.show()
