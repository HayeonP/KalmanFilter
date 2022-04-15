import rosbag
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import numpy as np
import copy

def create_inputs(data):
    inputs = []
    for i in range(len(data[0])):
        inputs.append([data[0][i], data[1][i], data[2][i], data[3][i]])

    return inputs 

def kalman_filter(data):
    outputs = []


    # prediction
    # mu0 = Hk xk
    # sigma0 = Hk Pk HTk

    return outputs

def main():
    bag = rosbag.Bag('files/220203_FMTC_ndt_pose_with_gps_data_red_course.bag')

    pose_list = []
    time_list = []

    for topic, msg, t in bag.read_messages(topics=['/ndt_pose']):
        pose_list.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        time_list.append(msg.header.stamp.secs+msg.header.stamp.nsecs/1000000000)

    vel_x_list = []
    vel_y_list = []

    for i in range(1,len(pose_list)):
        vel_x = (pose_list[i][0] - pose_list[i-1][0])/(time_list[i] - time_list[i-1])
        vel_y = (pose_list[i][1] - pose_list[i-1][1])/(time_list[i] - time_list[i-1])
        vel_x_list.append(vel_x)
        vel_y_list.append(vel_y)


    acc_x_list = []
    acc_y_list = []

    for i in range(1, len(vel_x_list)):
        acc_x = (vel_x_list[i] - vel_x_list[i-1])/(time_list[i+1] - time_list[i])
        acc_y = (vel_y_list[i] - vel_y_list[i-1])/(time_list[i+1] - time_list[i])
        acc_x_list.append(acc_x)
        acc_y_list.append(acc_y)

    # Align index of data structures
    pose_x_list = [pose[0] for pose in pose_list]
    pose_x_list = pose_x_list[2:]
    pose_y_list = [pose[1] for pose in pose_list]
    pose_y_list = pose_y_list[2:]
    vel_x_list = vel_x_list[1:]
    vel_y_list = vel_y_list[1:]

    time_list = time_list[2:]

    # initialization
    pose_x = pose_x_list[0]
    pose_y = pose_y_list[0]
    vel_x = vel_x_list[0]
    vel_y = vel_y_list[0]

    x_hat_prime_k = np.array([[pose_x], [pose_y], [vel_x], [vel_y]])
    P_prime_k = np.identity(4) # Initialize covariance as Identical matrix

    H_k = np.array([[1,      0,      0,      0],
                    [0,      1,      0,      0],
                    [0,      0,      1,      0],
                    [0,      0,      0,      1]])

    # Q_k = np.array([[1,      0,      0,      0],
    #                 [0,      1,      0,      0],
    #                 [0,      0,      1,      0],
    #                 [0,      0,      0,      1]])


    Q_k = np.array([[2,      0,      1,      0],
                    [0,      2,      0,      1],
                    [1,      0,      2,      0],
                    [0,      1,      0,      2]])
    
    R_k = np.array([[1,      0,      0,      0],
                    [0,      1,      0,      0],
                    [0,      0,      1,      0],
                    [0,      0,      0,      1]])

    # Output
    output_pose_x_list = []
    output_pose_y_list = []

    for i in range(1, len(pose_x_list)):
        x_hat_k_1 = copy.deepcopy(x_hat_prime_k)
        P_k_1 = copy.deepcopy(P_prime_k)
        theta_t = time_list[i] - time_list[i-1]

        # Predict
        F_k = np.array([[1,     0,      theta_t, 0      ],
                        [0,     1,      0,      theta_t ],
                        [0,     0,      1,      0       ],
                        [0,     0,      0,      1       ]])
        
        theta_t_2 = pow(theta_t, 2)
        B_k = np.array([[0.5*theta_t_2,     0,              0,              0       ],
                        [0,                 0.5*theta_t_2,  0,              0       ],
                        [0,                 0,              theta_t,        0       ],
                        [0,                 0,              0,              theta_t ]])

        acc_x_k_1 = acc_x_list[i-1]
        acc_y_k_1 = acc_y_list[i-1]
        u_k = np.array([[acc_x_k_1], [acc_y_k_1], [acc_x_k_1], [acc_y_k_1]])

        x_hat_k = F_k @ x_hat_k_1 + B_k @ u_k
        P_k = F_k @ P_k_1 @ np.transpose(F_k) + Q_k
        
        output_pose_x_list.append(x_hat_k[0])
        output_pose_y_list.append(x_hat_k[1])

        # Update

        pose_x = pose_x_list[i]
        pose_y = pose_y_list[i]
        vel_x = vel_x_list[i]
        vel_y = vel_y_list[i]
        z_k = np.array([[pose_x], [pose_y], [vel_x], [vel_y]])

        K_prime = P_k @ H_k @ np.linalg.pinv(H_k @ P_k @ np.transpose(H_k) + R_k)

        x_hat_prime_k = x_hat_k + K_prime @ (z_k - H_k @ x_hat_k)
        P_prime_k = P_k - K_prime @ H_k @ P_k
    


    plt.plot(pose_x_list, pose_y_list, linewidth=0.5, label='origin')
    plt.plot(output_pose_x_list, output_pose_y_list, linewidth=0.5, label='KF')
    plt.legend()
    plt.show()
    plt.close()


    return

if __name__ == '__main__':
    main()

'''
header: 
  seq: 10449
  stamp: 
    secs: 1643870365
    nsecs: 239412000
  frame_id: "/map"
pose: 
  position: 
    x: 87.12862396240234
    y: 133.9093017578125
    z: 6.892755031585693
  orientation: 
    x: 0.06568801780013588
    y: 0.0318829475407855
    z: -0.00801276376061509
    w: 0.9972985298247045

'''