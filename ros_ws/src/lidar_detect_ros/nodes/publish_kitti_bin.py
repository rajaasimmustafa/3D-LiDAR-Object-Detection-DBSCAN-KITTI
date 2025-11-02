#!/usr/bin/env python3
import rospy
import sys
import os
import numpy as np
import struct
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

def read_bin(path):
    """Read KITTI .bin file -> Nx4 array (x,y,z,intensity)."""
    scan = np.fromfile(path, dtype=np.float32)
    if scan.size % 4 != 0:
        raise ValueError("Invalid .bin file (not divisible by 4): " + path)
    return scan.reshape(-1, 4)

def create_cloud_xyz_ir(header, points):
    """
    Create a sensor_msgs/PointCloud2 message with fields x,y,z,intensity.
    'points' is Nx4 numpy array.
    """
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
    ]
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16  # 4 floats * 4 bytes
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = False

    # pack all points into bytes
    buff = bytearray()
    for p in points:
        buff.extend(struct.pack('ffff', float(p[0]), float(p[1]), float(p[2]), float(p[3])))
    msg.data = bytes(buff)
    return msg

def publish_once(file_path, pub, frame_id="velodyne"):
    pts = read_bin(file_path)
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    pc2_msg = create_cloud_xyz_ir(header, pts)
    pub.publish(pc2_msg)
    rospy.loginfo("Published %d points from %s", pts.shape[0], os.path.basename(file_path))
   

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: rosrun my_lidar_processing publish_kitti_bin.py <path-to-bin> [rate_hz]")
        sys.exit(1)

    file_path = os.path.expanduser(sys.argv[1])
    if not os.path.isfile(file_path):
        print("File not found:", file_path)
        sys.exit(1)

    rate_hz = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    rospy.init_node("kitti_bin_publisher", anonymous=True)
    pub = rospy.Publisher("/velodyne_points", PointCloud2, queue_size=1, latch=True)

    rospy.sleep(0.5)  # wait for subscribers

    try:
        publish_once(file_path, pub, frame_id="velodyne")  # ✅ publish one frame
        rospy.loginfo("Publishing complete. Exiting...")
    except rospy.ROSInterruptException:
        pass

