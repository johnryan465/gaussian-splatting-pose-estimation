from nptyping import NDArray, Float, Shape
import numpy as np


def J_term(t: NDArray[Shape["3,1"], Float]) -> NDArray[Shape["2,3"], Float]:
    return np.array(
        [
            [1/t[2, 0], 0, -t[0, 0]/(t[2, 0]**2)],
            [0, 1/t[2, 0], -t[1, 0]/(t[2, 0]**2)]
        ]
    )


def m_term(t: NDArray[Shape["3,1"], Float]) -> NDArray[Shape["2,1"], Float]:
    return np.array(
        [
            [t[0, 0]/t[2, 0]],
            [t[1, 0]/t[2, 0]]
        ]
    )


def grad_mu_d(t: NDArray[Shape["3,1"], Float]) -> NDArray[Shape["3,2,1"], Float]:
    return np.array([
        [-1/t[2, 0], 0],
        [0, -1/t[2, 0]],
        [t[0, 0]/(t[2, 0] * t[2, 0]), t[1, 0]/(t[2, 0] * t[2, 0])]
    ])


def grad_J_d(t: NDArray[Shape["3,1"], Float]) -> NDArray[Shape["3,2,3"], Float]:
    return np.array([
        [[0, 0, -1/(t[2, 0]*t[2, 0])], [0, 0, 0]],
        [[0, 0, 0], [0, 0, -1/(t[2, 0]*t[2, 0])]],
        [[-1/(t[2, 0]*t[2, 0]), 0, (2*t[0, 0])/(t[2, 0]*t[2, 0]*t[2, 0])], [0, -1/(t[2, 0]*t[2, 0]), (2*t[1, 0])/(t[2, 0]*t[2, 0]*t[2, 0])]]])

def exp_term(x: NDArray[Shape["2,1"], Float], u: NDArray[Shape["3,1"], Float], sigma_3d: NDArray[Shape["3,3"], Float], w: NDArray[Shape["3,3"], Float], d: NDArray[Shape["3,1"], Float]) -> NDArray[Shape["1,1"], Float]:
    t = w @ u + d
    mu = x - m_term(t)
    J = J_term(t)
    sigma = J @ w @ sigma_3d @ w.T @ J.T
    return np.exp(-0.5 * mu.T @ np.linalg.inv(sigma) @ mu)


def exp_term_gradd(x: NDArray[Shape["2,1"], Float], u: NDArray[Shape["3,1"], Float], sigma_3d: NDArray[Shape["3,3"], Float], w: NDArray[Shape["3,3"], Float], d: NDArray[Shape["3,1"], Float]):
    # Used in forward pass
    t = w @ u + d
    mu = x - m_term(t)
    J = J_term(t)
    J_w_sigma_wt = J @ w @ sigma_3d @ w.T
    sigma = J_w_sigma_wt @ J.T
    sigma_inv = np.linalg.inv(sigma)
    sigma_inv_mu = sigma_inv @ mu

    e = np.exp(-0.5 * mu.T @ sigma_inv_mu)
    # New in backward pass
    grad_J = grad_J_d(t)
    grad_mu = grad_mu_d(t)
    res = (sigma_inv_mu.T @ J_w_sigma_wt)[None, :, :] @ grad_J.transpose(0, 2, 1) @ (sigma_inv_mu)[None, :, :]
    res = res.squeeze(-1)
    res = res - grad_mu @ sigma_inv_mu

    return res * e




if __name__ == "__main__":
    x = np.array([[0], [0]])
    u = np.array([[1], [2], [3]])
    sigma_3d = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
    w = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    d = np.array([[1], [2], [3]])
    print(exp_term(x, u, sigma_3d, w, d))
    print(exp_term_gradd(x, u, sigma_3d, w, d))
