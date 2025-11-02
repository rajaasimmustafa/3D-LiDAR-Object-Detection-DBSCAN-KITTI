#!/usr/bin/env python3
import os, sys, glob, struct, time
import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

def read_bin(path):
    scan = np.fromfile(path, dtype=np.float32)
    if scan.size % 4 != 0:
        raise ValueError("Invalid .bin file (not divisible by 4): " + path)
    return scan.reshape(-1, 4)

def create_cloud_xyz_ir(header, points):
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
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = False

    buff = bytearray()
    for p in points:
        buff.extend(struct.pack('ffff', float(p[0]), float(p[1]), float(p[2]), float(p[3])))
    msg.data = bytes(buff)
    return msg

def publish_file(file_path, pub, frame_id="velodyne"):
    pts = read_bin(file_path)
    header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    pc2_msg = create_cloud_xyz_ir(header, pts)
    pub.publish(pc2_msg)
    rospy.loginfo("Published %d points from %s", pts.shape[0], os.path.basename(file_path))

if __name__ == "__main__":
    rospy.init_node("kitti_bin_publisher", anonymous=True)

    # --- Robust argument/param handling ---
    # Prefer ROS params; fall back to argv (but ignore ROS extras like __name:=...)
    args = rospy.myargv(argv=sys.argv)

    data_dir  = rospy.get_param("~data_dir", None)     # folder of .bin
    file_path = rospy.get_param("~file_path", None)    # single .bin
    rate_hz   = float(rospy.get_param("~rate_hz", 1.0))
    loop_play = bool(rospy.get_param("~loop", False))
    frame_id  = rospy.get_param("~frame_id", "velodyne")
    topic     = rospy.get_param("~topic", "/velodyne_points")

    # Backward-compat: allow argv[1] as file or dir if params not given
    if (not data_dir and not file_path) and len(args) > 1:
        cand = os.path.expanduser(args[1])
        if os.path.isdir(cand):
            data_dir = cand
        else:
            file_path = cand

    if not data_dir and not file_path:
        rospy.logfatal("Provide either ~data_dir (dir with .bin) OR ~file_path (single .bin).")
        sys.exit(1)

    pub = rospy.Publisher(topic, PointCloud2, queue_size=1, latch=True)
    rate = rospy.Rate(rate_hz)

    try:
        if file_path:
            if not os.path.isfile(file_path):
                rospy.logfatal("File not found: %s", file_path)
                sys.exit(1)
            # publish single frame once
            rospy.sleep(0.3)
            publish_file(file_path, pub, frame_id)
            rospy.loginfo("Single frame published. Exiting.")
        else:
            # directory mode: stream all .bin files
            pattern = os.path.join(os.path.expanduser(data_dir), "*.bin")
            files = sorted(glob.glob(pattern))
            if not files:
                rospy.logfatal("No .bin files under: %s", data_dir)
                sys.exit(1)

            rospy.loginfo("Streaming %d frames from %s at %.2f Hz (loop=%s)",
                          len(files), data_dir, rate_hz, str(loop_play))

            while not rospy.is_shutdown():
                for fp in files:
                    publish_file(fp, pub, frame_id)
                    rate.sleep()
                if not loop_play:
                    rospy.loginfo("Done streaming all frames.")
                    break

    except rospy.ROSInterruptException:
        pass
