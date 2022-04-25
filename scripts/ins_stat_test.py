import numpy as np
import matplotlib.pyplot as plt
from rubis_msgs.msg import InsStat
import rosbag
import copy

class InsStatMsg:
    def __init__(self, msg):
        self.time = msg.header.stamp.secs+msg.header.stamp.nsecs/1000000000
        self.x_vel = msg.x_vel
        self.y_vel = msg.y_vel
        self.x_acc = msg.x_acc
        self.y_acc = msg.y_acc
        self.angular_velocity = msg.angular_velocity
        self.yaw = msg.yaw

        return

def main():
    pose_bag = rosbag.Bag('../files/220425_move_pose_gnss_tf.bag')
    ins_stat_bag = rosbag.Bag('../files/220425_move_pose_ins_stat.bag')

    ndt_noise_flag = True # Add noise to ndt pose data

    pose_list = []
    time_list = []

    for topic, msg, t in pose_bag.read_messages(topics=['/ndt_pose']):
        pose_list.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        time_list.append(msg.header.stamp.secs+msg.header.stamp.nsecs/1000000000)
    
    plt.plot([pose[0] for pose in pose_list], [pose[1] for pose in pose_list], linewidth=0.5, label='ndt_pose')
    

    raw_ins_stat_list = []
    for topic, msg, t in ins_stat_bag.read_messages(topics=['/ins_stat']):
        raw_ins_stat_list.append(InsStatMsg(msg))
    
    init_pose = [[pose_list[0][0]], [pose_list[0][1]]]


    ins_pose_list = []
    prev_pose = [init_pose]
    prev_time = raw_ins_stat_list[0].time

    raw_ins_stat_list = raw_ins_stat_list[1:]

    ins_stat_list = []

    idx = 1

    for time in time_list:
        for i in range(idx, len(raw_ins_stat_list)):
            stat = raw_ins_stat_list[i]
            if(stat.time > time):
                idx = i - 1
                break
        ins_stat_list.append(raw_ins_stat_list[idx])
    


    # Kalman Filter
    x_hat_prime_k = np.array([init_pose[0], init_pose[1]])

    P_prime_k = np.identity(2) # Initialize covariance as Identical matrix

    # Scale Matrix
    H_k = np.array([[1,      0],
                    [0,      1]])

    # Prediction Noise
    Q_I = 3
    Q_k = np.array([[Q_I,      0],
                    [0,      Q_I]])
    
    # Observation Noise
    R_I = 10
    R_k = np.array([[R_I,      0],
                    [0,      R_I]])

    # Output
    output_pose_x_list = []
    output_pose_y_list = []

    for i in range(1, len(time_list)):
        x_hat_k_prev = copy.deepcopy(x_hat_prime_k)
        P_k_prev = copy.deepcopy(P_prime_k)
        theta_t = time_list[i] - time_list[i-1]

        # Predict
        F_k = np.array([[1,     0],
                        [0,     1]])
        
        B_k = np.array([[theta_t,       0         ], 
                        [0,             theta_t   ]])

        u_k = np.array([[stat.x_vel + 0.5 * stat.x_acc * theta_t], [stat.y_vel + 0.5 * stat.y_acc * theta_t]])

        x_hat_k = F_k @ x_hat_k_prev + B_k @ u_k
        P_k = F_k @ P_k_prev @ np.transpose(F_k) + Q_k

        pose_x = pose_list[i][0]
        pose_y = pose_list[i][1]

        z_k = np.array([[pose_x], [pose_y]])

        K_prime = P_k @ H_k @ np.linalg.pinv(H_k @ P_k @ np.transpose(H_k) + R_k)

        x_hat_prime_k = x_hat_k + K_prime @ (z_k - H_k @ x_hat_k)
        P_prime_k = P_k - K_prime @ H_k @ P_k

        output_pose_x_list.append(x_hat_prime_k[0])
        output_pose_y_list.append(x_hat_prime_k[1])


    plt.plot(output_pose_x_list, output_pose_y_list, linewidth=0.5, label='KF')
    plt.legend()
    plt.show()


    return

if __name__ == '__main__':
    main()