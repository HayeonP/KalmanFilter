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

    pose_list = []

    for topic, msg, t in pose_bag.read_messages(topics=['/ndt_pose']):
        pose_list.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        # time_list.append(msg.header.stamp.secs+msg.header.stamp.nsecs/1000000000)
    
    plt.plot([pose[0] for pose in pose_list], [pose[1] for pose in pose_list], linewidth=0.5, label='ndt_pose')
    

    ins_stat_list = []
    for topic, msg, t in ins_stat_bag.read_messages(topics=['/ins_stat']):
        ins_stat_list.append(InsStatMsg(msg))
    init_pose = [[pose_list[0][0]], [pose_list[0][1]]]

    ins_pose_list = []
    prev_pose = [init_pose]
    prev_time = ins_stat_list[0].time

    ins_stat_list = ins_stat_list[1:]

    print(init_pose)    

    cnt = 0
    for stat in ins_stat_list:
        cur_time = copy.deepcopy(stat.time)
        theta_t = cur_time - prev_time

        B_k = np.array([[theta_t,       0         ], 
                        [0,             theta_t   ]])

        u_k = np.array([[stat.x_vel + 0.5 * stat.x_acc * theta_t], [stat.y_vel + 0.5 * stat.y_acc * theta_t]])
        cur_pose =  prev_pose + B_k @ u_k

        print(stat.x_vel, stat.y_vel, cur_pose)

        ins_pose_list.append(cur_pose[0])

        prev_pose = copy.deepcopy(cur_pose)
        prev_time = copy.deepcopy(cur_time)

        # cnt = cnt + 1
        # print(cur_pose)
        # if cnt > 100: break

    plt.plot([pose[0] for pose in ins_pose_list], [pose[1] for pose in ins_pose_list])
    plt.legend()
    plt.show()


    return

if __name__ == '__main__':
    main()