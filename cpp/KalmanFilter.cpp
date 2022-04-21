#include <iostream>
#include "KalmanFilter.hpp"

int main(int argc, char* argv[]){

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

    Eigen::Vector4f x_hat_k_prev; // Previous Value
    x_hat_k_prev <<     87.16960907,
                        133.9387207,
                        0.32146015,
                        0.24648557;


    //// Updated variables

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

    float acc_x_k_prev = 2.344874339221101;
    float acc_y_k_prev = 1.9954219601903327;

    Eigen::Vector4f u_k; // Control Vector *
    u_k <<  acc_x_k_prev,   acc_y_k_prev,   acc_x_k_prev,   acc_y_k_prev;

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

/*
[[ 87.16960907]     // pose_x
 [133.9387207 ]     // pose_y
 [  0.32146015]     // vel_x
 [  0.24648557]]    // vel_y
 */