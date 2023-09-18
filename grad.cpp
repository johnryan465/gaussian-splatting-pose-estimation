#include <iostream>
#include <cmath>

#include <glm/glm.hpp>


glm::mat3x2 J_term(glm::vec3 t){
        glm::mat3x2 J;
        J[0][0] = 1.0/t[2];
        J[1][0] = 0.0;
        J[2][0] = -t[0]/(t[2]*t[2]);
        J[0][1] = 0.0;
        J[1][1] = 1.0/t[2];
        J[2][1] = -t[1]/(t[2]*t[2]);
        return J;
}

glm::vec2 m_term(glm::vec3 t){
    glm::vec2 m;
    m[0] = t[0]/t[2];
    m[1] = t[1]/t[2];
    return m;
}

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


glm::mat3x2 grad_J_d_0(glm::vec3 t){
    glm::mat3x2 grad;
    grad[0][0] = 0.0;
    grad[1][0] = 0.0;
    grad[2][0] = -1.0/(t[2]*t[2]);
    grad[0][1] = 0.0;
    grad[1][1] = 0.0;
    grad[2][1] = 0.0;
    return grad;
}

glm::mat3x2 grad_J_d_1(glm::vec3 t){
    glm::mat3x2 grad;
    grad[0][0] = 0.0;
    grad[1][0] = 0.0;
    grad[2][0] = 0.0;
    grad[0][1] = 0.0;
    grad[1][1] = 0.0;
    grad[2][1] = -1.0/(t[2]*t[2]);
    return grad;
}

glm::mat3x2 grad_J_d_2(glm::vec3 t){
    glm::mat3x2 grad;
    grad[0][0] = -1.0/(t[2]*t[2]);
    grad[1][0] = 0.0;
    grad[2][0] = 2.0*t[0]/(t[2]*t[2]*t[2]);
    grad[0][1] = 0.0;
    grad[1][1] = -1.0/(t[2]*t[2]);
    grad[2][1] = 2.0*t[1]/(t[2]*t[2]*t[2]);
    return grad;
}

glm::vec3 exp_term_gradd(glm::vec2 x, glm::vec3 u, glm::mat3 sigma_3d, glm::mat3 w, glm::vec3 d){
    glm::vec3 t = w * u + d;
    glm::vec2 mu = x - m_term(t);
    glm::mat3x2 J = J_term(t);
    glm::mat3x2 J_w_sigma_wt = J * (w * sigma_3d * glm::transpose(w));

    glm::mat2 sigma = J_w_sigma_wt * glm::transpose(J);
    glm::mat2 sigma_inv = glm::inverse(sigma);
    glm::vec2 sigma_inv_mu = sigma_inv * mu;

    float e = std::exp(-0.5 * glm::dot(mu, sigma_inv_mu));
    glm::mat3x2 grad_J_0 = grad_J_d_0(t);
    glm::mat3x2 grad_J_1 = grad_J_d_1(t);
    glm::mat3x2 grad_J_2 = grad_J_d_2(t);

    glm::mat3x2 grad_mu = grad_mu_d(t);

    float res1 = glm::dot(sigma_inv_mu * J_w_sigma_wt, glm::transpose(grad_J_0) * sigma_inv_mu);
    glm::vec2 tmp1 = glm::vec2(grad_mu[0][0], grad_mu[0][1]);
    res1 = res1 - glm::dot(tmp1, sigma_inv_mu);

    float res2 = glm::dot(sigma_inv_mu * J_w_sigma_wt, glm::transpose(grad_J_1) * sigma_inv_mu);
    glm::vec2 tmp2 = glm::vec2(grad_mu[1][0], grad_mu[1][1]);
    res2 = res2 - glm::dot(tmp2, sigma_inv_mu);

    float res3 = glm::dot(sigma_inv_mu * J_w_sigma_wt, glm::transpose(grad_J_2) * sigma_inv_mu);
    glm::vec2 tmp3 = glm::vec2(grad_mu[2][0], grad_mu[2][1]);
    res3 = res3 -  glm::dot(tmp3, sigma_inv_mu);

    return glm::vec3(res1, res2, res3) * e;
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
