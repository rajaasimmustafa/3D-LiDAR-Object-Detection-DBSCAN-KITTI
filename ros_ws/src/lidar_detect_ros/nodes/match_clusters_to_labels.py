#!/usr/bin/env python3
import rospy, sys, os
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
import subprocess

# ---------------------------------------------------------
# Helper: Load KITTI Ground Truth (convert to Velodyne coords)
# ---------------------------------------------------------
def load_gt(label_file, calib_file):
    """
    Converts KITTI label_2 file into Velodyne coordinates
    using kitti_label_to_velo.py (subprocess call).
    Returns: list of (class_name, np.array([x, y, z])).
    """
    cmd = f"rosrun my_lidar_processing kitti_label_to_velo.py {calib_file} {label_file}"
    try:
        out = subprocess.check_output(cmd, shell=True).decode("utf-8").strip().split("\n")
    except subprocess.CalledProcessError as e:
        rospy.logerr("Failed to run label conversion: %s", e)
        return []

    gt_objects = []
    for line in out:
        if not line.strip():
            continue
        parts = line.split()
        obj_type = parts[0]
        if obj_type.lower() == "dontcare":
            continue  # Skip DontCare boxes

        try:
            coords = [float(x.strip("(),")) for x in parts[-3:]]
            gt_objects.append((obj_type, np.array(coords)))
        except:
            continue
    return gt_objects


# ---------------------------------------------------------
# Node: Match clusters to ground-truth
# ---------------------------------------------------------
class ClusterMatcher:
    def __init__(self, calib_file, label_file):
        rospy.init_node("realtime_cluster_matcher", anonymous=True)

        self.pub_marker = rospy.Publisher("/matched_gt_marker", Marker, queue_size=10)
        self.THRESH = rospy.get_param("~match_threshold", 3.0)  # m — small distance for KITTI frame

        # Load GT data
        self.gt_objects = load_gt(label_file, calib_file)
        if not self.gt_objects:
            rospy.logwarn("No GT objects loaded!")
        else:
            rospy.loginfo("Loaded %d GT objects:", len(self.gt_objects))
            for t, p in self.gt_objects:
                rospy.loginfo("  %s at %s", t, p)

        # Subscribe to cluster centroids
        rospy.Subscriber("/pcl_centroids", PointCloud2, self.centroid_callback, queue_size=1)
        rospy.loginfo("Subscribed to /pcl_centroids ... ready to match!")

    def centroid_callback(self, msg):
        centroids = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            centroids.append(np.array([p[0], p[1], p[2]]))
        if not centroids:
            rospy.loginfo_throttle(5, "No centroids to match.")
            return

        matched_any = False
        best_match_distance = {i: float('inf') for i, _ in enumerate(self.gt_objects)}
        best_match_centroid = {i: None for i, _ in enumerate(self.gt_objects)}

        # Find nearest centroid for each GT object
        for i, (gt_type, gt_pos) in enumerate(self.gt_objects):
            for cpos in centroids:
                dist = np.linalg.norm(cpos - gt_pos)
                if dist < self.THRESH and dist < best_match_distance[i]:
                    best_match_distance[i] = dist
                    best_match_centroid[i] = cpos
                    matched_any = True

        # Publish only matched GTs
        for i, (gt_type, gt_pos) in enumerate(self.gt_objects):
            best_dist = best_match_distance[i]
            if best_dist < float('inf'):
                print(f"Matched {gt_type} (distance = {best_dist:.2f} m)")
                cpos = best_match_centroid[i]

                # --- RViz marker for matched object ---
                m = Marker()
                m.header = Header()
                m.header.stamp = rospy.Time.now()
                m.header.frame_id = "velodyne"
                m.ns = "matched_gt"
                m.id = int(np.random.randint(0, 10000))
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(cpos[0])
                m.pose.position.y = float(cpos[1])
                m.pose.position.z = float(cpos[2])
                m.pose.orientation.w = 1.0
                m.scale.x = 0.7
                m.scale.y = 0.7
                m.scale.z = 0.7
                # yellow sphere
                m.color.r = 1.0
                m.color.g = 1.0
                m.color.b = 0.0
                m.color.a = 1.0
                self.pub_marker.publish(m)

        if not matched_any:
            rospy.loginfo_throttle(5, "No matches found in this frame.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: rosrun my_lidar_processing match_clusters_to_labels.py <calib.txt> <label.txt>")
        sys.exit(1)

    calib_file = sys.argv[1]
    label_file = sys.argv[2]

    if not os.path.isfile(calib_file):
        print("Calibration file not found:", calib_file)
        sys.exit(1)
    if not os.path.isfile(label_file):
        print("Label file not found:", label_file)
        sys.exit(1)

    matcher = ClusterMatcher(calib_file, label_file)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
