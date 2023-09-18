#include <iostream>
#include <cmath>

#include <glm/glm.hpp>

/*
def J_term(t: NDArray[Shape["3,1"], Float]) -> NDArray[Shape["2,3"], Float]:
    return np.array(
        [
            [1/t[2, 0], 0, -t[0, 0]/(t[2, 0]**2)],
            [0, 1/t[2, 0], -t[1, 0]/(t[2, 0]**2)]
        ]
    )
*/

glm::mat3x2 J_term(glm::vec3 t){
        glm::mat3x2 J;
        J[0][0] = 1.0/t[2];
        J[2][0] = -t[0]/(t[2]*t[2]);
        J[1][1] = 1.0/t[2];
        J[2][1] = -t[1]/(t[2]*t[2]);
        return J;
}

/*def m_term(t: NDArray[Shape["3,1"], Float]) -> NDArray[Shape["2,1"], Float]:
    return np.array(
        [
            [t[0, 0]/t[2, 0]],
            [t[1, 0]/t[2, 0]]
        ]
    )*/


glm::vec2 m_term(glm::vec3 t){
    glm::vec2 m;
    m[0] = t[0]/t[2];
    m[1] = t[1]/t[2];
    return m;
}

/*
def grad_mu_d(t: NDArray[Shape["3,1"], Float]) -> NDArray[Shape["3,2,1"], Float]:
    return np.array([
        [-1/t[2, 0], 0],
        [0, -1/t[2, 0]],
        [t[0, 0]/(t[2, 0] * t[2, 0]), t[1, 0]/(t[2, 0] * t[2, 0])]
    ])

*/

glm::mat3x2 grad_mu_d(glm::vec3 t){
    glm::mat3x2 grad;
    grad[0][0] = -1.0/t[2];
    grad[0][1] = 0.0;
    grad[1][0] = 0.0;
    grad[1][1] = -1.0/t[2];
    grad[2][0] = t[0]/(t[2]*t[2]);
    grad[2][1] = t[1]/(t[2]*t[2]);
    return grad;
}

/*
def grad_J_d(t: NDArray[Shape["3,1"], Float]) -> NDArray[Shape["3,2,3"], Float]:
    return np.array([
        [[0, 0, -1/(t[2, 0]*t[2, 0])], [0, 0, 0]],
        [[0, 0, 0], [0, 0, -1/(t[2, 0]*t[2, 0])]],
        [[-1/(t[2, 0]*t[2, 0]), 0, (2*t[0, 0])/(t[2, 0]*t[2, 0]*t[2, 0])], [0, -1/(t[2, 0]*t[2, 0]), (2*t[1, 0])/(t[2, 0]*t[2, 0]*t[2, 0])]]])

*/


glm::mat3x2 grad_J_d_0(glm::vec3 t){
    glm::mat3x2 grad;
    grad[2][0] = -1.0/(t[2]*t[2]);
    return grad;
}

glm::mat3x2 grad_J_d_1(glm::vec3 t){
    glm::mat3x2 grad;
    grad[2][1] = -1.0/(t[2]*t[2]);
    return grad;
}

glm::mat3x2 grad_J_d_2(glm::vec3 t){
    glm::mat3x2 grad;
    grad[0][0] = -1.0/(t[2]*t[2]);
    grad[2][0] = 2.0*t[0]/(t[2]*t[2]*t[2]);
    grad[1][1] = -1.0/(t[2]*t[2]);
    grad[2][1] = 2.0*t[1]/(t[2]*t[2]*t[2]);
    return grad;
}


/*
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
*/

glm::vec3 exp_term_gradd(glm::vec2 x, glm::vec3 u, glm::mat3 sigma_3d, glm::mat3 w, glm::vec3 d){
    glm::vec3 t = w * u + d;
    std::cout << "T:" << t[0] << ' ' << t[1] << ' ' << t[2] << std::endl;
    glm::vec2 mu = x - m_term(t);
    std::cout << mu[0] << ' ' << mu[1] << std::endl;
    glm::mat3x2 J = J_term(t);
    //std::cout << "J: " << J[0][0] << ' ' << J[0][1] << std::endl;
    //std::cout << "J: " << J[1][0] << ' ' << J[1][1] << std::endl;
    //std::cout << "J: " << J[2][0] << ' ' << J[2][1] << std::endl;
    glm::mat3x2 J_w_sigma_wt = J * (w * sigma_3d * glm::transpose(w));
    std::cout << "Jw: " << J_w_sigma_wt[0][0] << ' ' << J_w_sigma_wt[1][0] << ' ' << J_w_sigma_wt[2][0] << std::endl; 
    std::cout << "Jw: " << J_w_sigma_wt[0][1] << ' ' << J_w_sigma_wt[1][1] << ' ' << J_w_sigma_wt[2][1] << std::endl;

    glm::mat2 sigma = J_w_sigma_wt * glm::transpose(J);
    glm::mat2 sigma_inv = glm::inverse(sigma);
    std::cout << "Sigma: " << sigma_inv[0][0] << ' ' << sigma_inv[0][1] << std::endl;
    std::cout << "Sigma: " << sigma_inv[1][0] << ' ' << sigma_inv[1][1] << std::endl;
    glm::vec2 sigma_inv_mu = sigma_inv * mu;

    float e = std::exp(-0.5 * glm::dot(mu, sigma_inv_mu));
    glm::mat3x2 grad_J_1 = grad_J_d_1(t);
    glm::mat3x2 grad_J_0 = grad_J_d_0(t);
    glm::mat3x2 grad_J_2 = grad_J_d_2(t);
    std::cout << "grad_J_0: " << grad_J_0[0][0] << ' ' << grad_J_0[1][0] << ' ' << grad_J_0[2][0] << std::endl;
    std::cout << "grad_J_0: " << grad_J_0[0][1] << ' ' << grad_J_0[1][1] << ' ' << grad_J_0[2][1] << std::endl;
    std::cout << "grad_J_1: " << grad_J_1[0][0] << ' ' << grad_J_1[1][0] << ' ' << grad_J_1[2][0] << std::endl;
    std::cout << "grad_J_1: " << grad_J_1[0][1] << ' ' << grad_J_1[1][1] << ' ' << grad_J_1[2][1] << std::endl;
    std::cout << "grad_J_2: " << grad_J_2[0][0] << ' ' << grad_J_2[1][0] << ' ' << grad_J_2[2][0] << std::endl;
    std::cout << "grad_J_2: " << grad_J_2[0][1] << ' ' << grad_J_2[1][1] << ' ' << grad_J_2[2][1] << std::endl;

    glm::mat3x2 grad_mu = grad_mu_d(t);
    std::cout << "grad_mu: " << grad_mu[0][0] << ' ' << grad_mu[0][1] << std::endl;
    std::cout << "grad_mu: " << grad_mu[1][0] << ' ' << grad_mu[1][1] << std::endl;
    std::cout << "grad_mu: " << grad_mu[2][0] << ' ' << grad_mu[2][1] << std::endl;
    // glm::mat2x3 res = glm::transpose(sigma_inv_mu) * J_w_sigma_wt * glm::transpose(sigma_inv_mu);
    //res = res - grad_mu * sigma_inv_mu;

    glm::vec3 test =  sigma_inv_mu * J_w_sigma_wt;
    std::cout << "test: " << test[0] << ' ' << test[1] << ' ' << test[2] << std::endl;
    glm::vec3 test2 = glm::transpose(grad_J_2) * sigma_inv_mu;
    std::cout << "test2: " << test2[0] << ' ' << test2[1] << ' ' << test2[2] << std::endl;

    std::cout << "sigma_inv_mu: " << sigma_inv_mu[0] << ' ' << sigma_inv_mu[1] << std::endl;
    glm::vec1 res1 = sigma_inv_mu * J_w_sigma_wt * glm::transpose(grad_J_0) * sigma_inv_mu;
    glm::vec2 tmp1 = glm::vec2(grad_mu[0][0], grad_mu[0][1]);
    std::cout << "res1: " <<  res1[0] << std::endl;
    res1 = res1 - glm::dot(tmp1, sigma_inv_mu);
    std::cout << res1[0] << std::endl;

    glm::vec1 res2 = sigma_inv_mu * J_w_sigma_wt * glm::transpose(grad_J_1) * sigma_inv_mu;
    glm::vec2 tmp2 = glm::vec2(grad_mu[1][0], grad_mu[1][1]);
    std::cout << "res2: " <<  res2[0] << std::endl;
    res2 = res2 - glm::dot(tmp2, sigma_inv_mu);
    std::cout << res2[0] << std::endl;

    glm::vec1 res3 = sigma_inv_mu * J_w_sigma_wt * glm::transpose(grad_J_2) * sigma_inv_mu;
    glm::vec2 tmp3 = glm::vec2(grad_mu[2][0], grad_mu[2][1]);
    std::cout << "res3: " <<  res3[0] << std::endl;
    res3 = res3 -  glm::dot(tmp3, sigma_inv_mu);
    std::cout << res3[0] << std::endl;


    return glm::vec3(res1[0], res2[0], res3[0]);
}


int main(){
    glm::vec2 x = glm::vec2(0.0, 0.0);
    glm::vec3 u = glm::vec3(1.0, 2.0, 3.0);
    glm::mat3 sigma_3d = glm::mat3(3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0, 1.0);
    glm::mat3 w = glm::mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0 , 0.0, 0.0, 1.0);
    glm::vec3 d = glm::vec3(1.0, 2.0, 3.0);
    glm::vec3 res = exp_term_gradd(x, u, sigma_3d, w, d);
    std::cout << res[0] << ' ' << res[1] << ' '<< res[2] << std::endl;
    return 0;
}
