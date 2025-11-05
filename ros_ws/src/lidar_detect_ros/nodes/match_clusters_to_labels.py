#!/usr/bin/env python3
import rospy, sys, os, math, subprocess
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion

# -----------------------------
# Globals (boxes + stamps)
# -----------------------------
_cluster_boxes = []        # each: {'cx','cy','sx','sy','sz','quat'}
_last_boxes_stamp = None   # ROS time (from first marker’s header if present)
_last_cent_stamp  = None   # ROS time from centroids msg.header.stamp

def _marker_array_cb(msg):
    """MarkerArray ka apna header nahi hota; per-marker headers se kaam lo."""
    global _cluster_boxes, _last_boxes_stamp
    boxes = []
    first_stamp = None
    for m in msg.markers:
        if m.type != Marker.CUBE:
            continue
        boxes.append({
            'cx': float(m.pose.position.x),
            'cy': float(m.pose.position.y),
            'sx': float(m.scale.x),
            'sy': float(m.scale.y),
            'sz': float(m.scale.z),
            'quat': (
                float(m.pose.orientation.x),
                float(m.pose.orientation.y),
                float(m.pose.orientation.z),
                float(m.pose.orientation.w),
            ),
        })
        if first_stamp is None:
            # per-marker header stamp (agar publisher ne set kiya ho)
            try:
                if hasattr(m, "header") and m.header.stamp and m.header.stamp != rospy.Time():
                    first_stamp = m.header.stamp
            except Exception:
                pass
    _cluster_boxes = boxes
    _last_boxes_stamp = first_stamp if first_stamp is not None else rospy.Time.now()

# -----------------------------
# Small helpers
# -----------------------------
def _aabb_from_center(cx, cy, sx, sy):
    hx, hy = max(0.05, sx*0.5), max(0.05, sy*0.5)
    return (cx - hx, cy - hy, cx + hx, cy + hy)

def _iou_aabb_2d(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / max(union, 1e-9)

def _yaw_from_quat(qx, qy, qz, qw):
    r, p, y = euler_from_quaternion([qx, qy, qz, qw])
    return y

def _read_dims_from_kitti(label_file):
    dims = []  # (type, h, w, l, rot_y)
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 15 and parts[0].lower() != "dontcare":
                typ = parts[0]
                h = float(parts[8]); w = float(parts[9]); l = float(parts[10])
                rot_y = float(parts[14])
                dims.append((typ, h, w, l, rot_y))
    return dims

# ---------------------------------------------------------
# Helper: Load KITTI Ground Truth (convert to Velodyne coords)
# ---------------------------------------------------------
def load_gt(label_file, calib_file):
    """Uses kitti_label_to_velo.py output + dims order assumption."""
    cmd = f"rosrun my_lidar_processing kitti_label_to_velo.py {calib_file} {label_file}"
    try:
        out = subprocess.check_output(cmd, shell=True).decode("utf-8").strip().split("\n")
    except subprocess.CalledProcessError as e:
        rospy.logerr("Label conversion failed.\nCommand: %s\nOutput:\n%s",
                     cmd, e.output.decode("utf-8") if e.output else str(e))
        return []
    dims_list = _read_dims_from_kitti(label_file)
    gt = []
    k = 0
    for line in out:
        if not line.strip():
            continue
        parts = line.split()
        obj_type = parts[0]
        if obj_type.lower() == "dontcare":
            continue
        try:
            coords = [float(x.strip("(),")) for x in parts[-3:]]
            if k < len(dims_list):
                typ, h, w, l, rot_y = dims_list[k]; k += 1
                lw = (l, w)
            else:
                lw = (1.2, 0.6)
            gt.append((obj_type, np.array(coords, dtype=float), lw))
        except Exception:
            continue
    return gt

# ---------------------------------------------------------
# Node: Match + LOG (per-frame overwrite)
# ---------------------------------------------------------
class ClusterMatcher:
    def __init__(self, calib_file, label_file):
        rospy.init_node("realtime_cluster_matcher", anonymous=True)

        self.pub_marker = rospy.Publisher("/matched_gt_marker", Marker, queue_size=10)

        # gates
        self.dist_max = float(rospy.get_param("~match_distance_max", 6.0))
        self.iou_min  = float(rospy.get_param("~match_iou_min", 0.05))
        self.wait_window = float(rospy.get_param("~sync_wait_sec", 1.2))
        self.force_if_skew = bool(rospy.get_param("~force_if_skew", True))

        # ids + logs
        self.kitti_id = os.path.splitext(os.path.basename(label_file))[0]
        self.log_dir = os.path.expanduser("~/det_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"{self.kitti_id}.txt")

        # Load GT
        self.gt_objects = load_gt(label_file, calib_file)
        if not self.gt_objects:
            rospy.logwarn("No GT objects loaded!")
        else:
            rospy.loginfo("Loaded %d GT objects:", len(self.gt_objects))
            for (t, p, lw) in self.gt_objects:
                rospy.loginfo("  %s at %s (l=%.2f, w=%.2f)", t, p, lw[0], lw[1])

        # subs
        rospy.Subscriber("/pcl_centroids", PointCloud2, self.centroid_callback, queue_size=1)
        rospy.Subscriber("lidar_bounding_boxes", MarkerArray, _marker_array_cb, queue_size=1)

        rospy.loginfo("Subscribed to /pcl_centroids and lidar_bounding_boxes ... ready to match!")

    def _append_line(self, lines, det_type, cpos, box, score):
        if box is not None:
            l = max(0.1, box['sx'])
            w = max(0.1, box['sy'])
            h = max(0.1, box['sz'])
            yaw = _yaw_from_quat(*box['quat'])
        else:
            l, w, h, yaw = 1.2, 0.6, 1.7, 0.0
        lines.append(f"{det_type} {cpos[0]:.3f} {cpos[1]:.3f} {cpos[2]:.3f} {l:.3f} {w:.3f} {h:.3f} {yaw:.6f} {score:.3f}")

    def centroid_callback(self, msg):
        global _last_cent_stamp

        # collect centroids
        centroids = []
        for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
            centroids.append(np.array([p[0], p[1], p[2]], dtype=float))

        # centroid stamp (prefer message stamp)
        _last_cent_stamp = msg.header.stamp if (hasattr(msg, "header") and msg.header.stamp) else rospy.Time.now()

        # wait-until-data: boxes present?
        if _last_boxes_stamp is None or len(_cluster_boxes) == 0:
            rospy.loginfo_throttle(2.0, "Waiting for lidar_bounding_boxes ...")
            # empty write => "no dets"
            open(self.log_path, "w").close()
            return

        # skew gate
        try:
            skew = abs((_last_cent_stamp - _last_boxes_stamp).to_sec())
        except Exception:
            skew = float("inf")
        rospy.loginfo_throttle(5.0, f"stamps: cent={_last_cent_stamp.to_sec():.3f}, boxes={_last_boxes_stamp.to_sec():.3f}, |Δ|={skew:.3f}s")
        if skew > self.wait_window and not self.force_if_skew:
            rospy.loginfo_throttle(2.0, f"Stamp skew {skew:.3f}s > sync_wait_sec={self.wait_window:.2f}s. Waiting...")
            open(self.log_path, "w").close()
            return

        if not centroids:
            rospy.loginfo_throttle(5, "No centroids to match.")
            open(self.log_path, "w").close()
            return

        incoming_frame_id = "velodyne"
        if hasattr(msg, "header") and msg.header.frame_id:
            incoming_frame_id = msg.header.frame_id

        # precompute box AABBs
        cluster_aabbs = [ _aabb_from_center(b['cx'], b['cy'], b['sx'], b['sy']) for b in _cluster_boxes ]
        used_box_idx = set()
        used_cent_idx = set()

        out_lines = []
        matched_any = False

        # per-GT: nearest centroid (not-yet-used), then best IoU box (not-yet-used)
        for gi, (gt_type, gt_pos, (gl, gw)) in enumerate(self.gt_objects):
            # choose nearest centroid that is not yet used
            best_c_idx, best_c, best_d = -1, None, float('inf')
            for ci, cpos in enumerate(centroids):
                if ci in used_cent_idx:
                    continue
                d = np.linalg.norm(cpos - gt_pos)
                if d < best_d:
                    best_d, best_c_idx, best_c = d, ci, cpos
            if best_c is None or best_d > self.dist_max:
                print(f"No match for {gt_type} (best_d = {best_d:.2f}, best_iou = 0.00)")
                continue

            # box IoU (AABB BEV)
            gt_aabb = _aabb_from_center(gt_pos[0], gt_pos[1], gl, gw)
            best_iou, best_bi = 0.0, -1
            for bi, ca in enumerate(cluster_aabbs):
                if bi in used_box_idx:
                    continue
                iou = _iou_aabb_2d(ca, gt_aabb)
                if iou > best_iou:
                    best_iou, best_bi = iou, bi

            if best_bi >= 0 and best_iou >= self.iou_min:
                used_cent_idx.add(best_c_idx)
                used_box_idx.add(best_bi)
                matched_any = True
                print(f"Matched {gt_type} (distance = {best_d:.2f} m, IoU = {best_iou:.2f})")

                # viz
                m = Marker()
                m.header = Header(stamp=rospy.Time.now(), frame_id=incoming_frame_id)
                m.ns = "matched_gt"; m.id = int(np.random.randint(0, 10000))
                m.type = Marker.SPHERE; m.action = Marker.ADD
                m.pose.position.x = float(best_c[0]); m.pose.position.y = float(best_c[1]); m.pose.position.z = float(best_c[2])
                m.pose.orientation.w = 1.0
                m.scale.x = 0.7; m.scale.y = 0.7; m.scale.z = 0.7
                m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0
                self.pub_marker.publish(m)

                chosen_box = _cluster_boxes[best_bi] if 0 <= best_bi < len(_cluster_boxes) else None
                self._append_line(out_lines, gt_type, best_c, chosen_box, score=float(best_iou))
            else:
                print(f"No match for {gt_type} (best_d = {best_d:.2f}, best_iou = {best_iou:.2f})")

        if not matched_any:
            rospy.loginfo_throttle(3.0, "No matches passed the gates in this frame.")

        # overwrite log (latest only)
        with open(self.log_path, "w") as f:
            if out_lines:
                f.write("\n".join(out_lines) + "\n")
            # else: empty file = no detections

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: rosrun my_lidar_processing match_clusters_to_labels.py <calib.txt> <label.txt>")
        sys.exit(1)
    calib_file = sys.argv[1]
    label_file = sys.argv[2]
    if not os.path.isfile(calib_file):
        print("Calibration file not found:", calib_file); sys.exit(1)
    if not os.path.isfile(label_file):
        print("Label file not found:", label_file); sys.exit(1)
    matcher = ClusterMatcher(calib_file, label_file)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
