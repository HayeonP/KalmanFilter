#include <Eigen/Core>
#include <Eigen/Dense>

class LKF{
public:
    LKF(float init_pose_x, float init_pose_y);
    Eigen::Vector2f run(float theta_t, Eigen::Vector2f u_k, Eigen::Vector2f z_k); 
    // u_k: control vector, z_k: observation vector
private:
    Eigen::Matrix2f H_k; // Scale Matrix
    Eigen::Matrix2f Q_k; // Prediction Noise
    Eigen::Matrix2f R_k; // Observation Noise
    Eigen::Vector2f x_hat_k_prev; // Previous Value
    Eigen::Matrix2f P_k_prev; // Previous Prediction Covariance Matrix
};

