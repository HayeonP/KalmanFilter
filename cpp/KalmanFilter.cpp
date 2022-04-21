#include <iostream>
#include "KalmanFilter.hpp"


int main(int argc, char* argv[]){
    LKF linear_kalman_filter(87.16960907, 133.9387207);

    Eigen::Vector4f u_k;
    float acc_x_k = 2.344874339221101;
    float acc_y_k = 1.9954219601903327;
    u_k <<  acc_x_k,   acc_y_k,   acc_x_k,   acc_y_k;

    Eigen::Vector4f z_k;
    z_k << 87.1461181640625, 133.94679260253906, -0.232910746, 0.0800323382;

    Eigen::Vector4f output = linear_kalman_filter.run(0.10085797309875488, u_k, z_k);

    return 0;
}

int test(){

    //// Unchanged variables

    Eigen::Matrix4f H_k; // Scale Matrix
    H_k <<      1.0f,       0.0f,       0.0f,       0.0f,
                0.0f,       1.0f,       0.0f,       0.0f,
                0.0f,       0.0f,       1.0f,       0.0f,
                0.0f,       0.0f,       0.0f,       1.0f;

    Eigen::Matrix4f Q_k; // Prediction Noise
    Q_k <<      1.0f,       0.0f,       0.0f,       0.0f,
                0.0f,       1.0f,       0.0f,       0.0f,
                0.0f,       0.0f,       1.0f,       0.0f,
                0.0f,       0.0f,       0.0f,       1.0f;

    Eigen::Matrix4f R_k; // Observation Noise
    R_k <<      1.0f,       0.0f,       0.0f,       0.0f,
                0.0f,       1.0f,       0.0f,       0.0f,
                0.0f,       0.0f,       1.0f,       0.0f,
                0.0f,       0.0f,       0.0f,       1.0f;


    //// Updated variables

    Eigen::Vector4f x_hat_k_prev; // Previous Value
    x_hat_k_prev <<     87.16960907,
                        133.9387207,
                        0.32146015,
                        0.24648557;

    Eigen::Matrix4f P_k_prev; // Previous prediction covariance matrix
    P_k_prev <<     1.0f,       0.0f,       0.0f,       0.0f,
                    0.0f,       1.0f,       0.0f,       0.0f,
                    0.0f,       0.0f,       1.0f,       0.0f,
                    0.0f,       0.0f,       0.0f,       1.0f;

    float theta_t = 0.10085797309875488; // *

    // Prediction
    Eigen::Matrix4f F_k; // Prediction Matrix
    F_k <<      1.0f,       0.0f,    theta_t,        0.0f,
                0.0f,       1.0f,       0.0f,     theta_t,
                0.0f,       0.0f,       1.0f,        0.0f,
                0.0f,       0.0f,       0.0f,        1.0f;
    
    Eigen::Matrix4f B_k;
    B_k <<     0.5f * pow(theta_t,2),                   0.0f,       0.0f,       0.0f,
                                0.0f,  0.5f * pow(theta_t,2),       0.0f,       0.0f,
                                0.0f,                   0.0f,    theta_t,       0.0f,
                                0.0f,                   0.0f,       0.0f,     theta_t;

    float acc_x_k = 2.344874339221101;
    float acc_y_k = 1.9954219601903327;

    Eigen::Vector4f u_k; // Control Vector *
    u_k <<  acc_x_k,   acc_y_k,   acc_x_k,   acc_y_k;

    Eigen::Vector4f x_hat_k; // Prediction Result
    x_hat_k = F_k * x_hat_k_prev + B_k * u_k;

    Eigen::Matrix4f P_k; // Prediction Covariance
    P_k = F_k * P_k_prev * F_k.transpose() + Q_k;
    
    // Update
    Eigen::Vector4f z_k; // Observation Vector *
    z_k << 87.1461181640625, 133.94679260253906, -0.232910746, 0.0800323382;

    Eigen::Matrix4f K_prime; // Kalman gain(Modified)
    K_prime = P_k * H_k * (H_k * P_k * H_k.transpose() + R_k).completeOrthogonalDecomposition().pseudoInverse();
    
    Eigen::Vector4f x_hat_prime_k; // Update result
    x_hat_prime_k = x_hat_k + K_prime * (z_k - H_k * x_hat_k);

    Eigen::Matrix4f P_prime_k; // Update prediction covariance
    P_prime_k = P_k - K_prime * H_k * P_k;

    return 0;
}

LKF::LKF(float init_pose_x, float init_pose_y){
    H_k <<      1.0f,       0.0f,       0.0f,       0.0f,
                0.0f,       1.0f,       0.0f,       0.0f,
                0.0f,       0.0f,       1.0f,       0.0f,
                0.0f,       0.0f,       0.0f,       1.0f;

    Q_k <<      1.0f,       0.0f,       0.0f,       0.0f,
                0.0f,       1.0f,       0.0f,       0.0f,
                0.0f,       0.0f,       1.0f,       0.0f,
                0.0f,       0.0f,       0.0f,       1.0f;

    R_k <<      1.0f,       0.0f,       0.0f,       0.0f,
                0.0f,       1.0f,       0.0f,       0.0f,
                0.0f,       0.0f,       1.0f,       0.0f,
                0.0f,       0.0f,       0.0f,       1.0f;

    x_hat_k_prev << init_pose_x,    init_pose_y,    0.0f,   0.0f;

    P_k_prev << 1.0f,       0.0f,       0.0f,       0.0f,
                0.0f,       1.0f,       0.0f,       0.0f,
                0.0f,       0.0f,       1.0f,       0.0f,
                0.0f,       0.0f,       0.0f,       1.0f;

}

Eigen::Vector4f LKF::run(float theta_t, Eigen::Vector4f u_k, Eigen::Vector4f z_k){ // z_k: observation vector
    // Prediction
    Eigen::Matrix4f F_k; // Prediction Matrix
    F_k <<      1.0f,       0.0f,    theta_t,        0.0f,
                0.0f,       1.0f,       0.0f,     theta_t,
                0.0f,       0.0f,       1.0f,        0.0f,
                0.0f,       0.0f,       0.0f,        1.0f;

    Eigen::Matrix4f B_k; // Control Matrix
    B_k <<     0.5f * pow(theta_t,2),                   0.0f,       0.0f,       0.0f,
                                0.0f,  0.5f * pow(theta_t,2),       0.0f,       0.0f,
                                0.0f,                   0.0f,    theta_t,       0.0f,
                                0.0f,                   0.0f,       0.0f,     theta_t;

    Eigen::Vector4f x_hat_k; // Predict Result
    x_hat_k = F_k * x_hat_k_prev + B_k * u_k;

    Eigen::Matrix4f P_k; // Prediction Covariance
    P_k = F_k * P_k_prev * F_k.transpose() + Q_k;

    // Update
    Eigen::Matrix4f K_prime; // Kalman gain(Modified)
    K_prime = P_k * H_k * (H_k * P_k * H_k.transpose() + R_k).completeOrthogonalDecomposition().pseudoInverse();

    Eigen::Vector4f x_hat_prime_k; // Update result
    x_hat_prime_k = x_hat_k + K_prime * (z_k - H_k * x_hat_k);

    Eigen::Matrix4f P_prime_k; // Update prediction covariance
    P_prime_k = P_k - K_prime * H_k * P_k;

    x_hat_k_prev = x_hat_prime_k;
    P_k_prev = P_prime_k;

    return x_hat_prime_k;
}



/*
[[ 87.16960907]     // pose_x
 [133.9387207 ]     // pose_y
 [  0.32146015]     // vel_x
 [  0.24648557]]    // vel_y
 */