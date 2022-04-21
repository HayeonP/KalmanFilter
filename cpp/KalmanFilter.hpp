#include <Eigen/Core>
#include <Eigen/Dense>

class LKF{
public:
    LKF(float init_pose_x, float init_pose_y);
    Eigen::Vector4f run(float theta_t, Eigen::Vector4f u_k, Eigen::Vector4f z_k); 
    // u_k: control vector, z_k: observation vector
private:
    Eigen::Matrix4f H_k; // Scale Matrix
    Eigen::Matrix4f Q_k; // Prediction Noise
    Eigen::Matrix4f R_k; // Observation Noise
    Eigen::Vector4f x_hat_k_prev; // Previous Value
    Eigen::Matrix4f P_k_prev; // Previous Prediction Covariance Matrix
};

