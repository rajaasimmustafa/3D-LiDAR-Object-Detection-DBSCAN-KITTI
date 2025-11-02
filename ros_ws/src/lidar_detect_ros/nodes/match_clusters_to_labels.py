#!/usr/bin/env python3
import rospy, sys, os, math, subprocess
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion

# -----------------------------
# Globals (boxes + timing flags)
# -----------------------------
_cluster_boxes = []                 # each: {'cx','cy','sx','sy','sz','quat'}
_last_boxes_stamp = None            # ROS time of last boxes
_last_centroids_stamp = None        # ROS time of last centroids
_USE_WALL_TIME_FOR_BOXES = True     # set from params in __init__
_FORCE_PROCESS_IF_SKEW   = True     # set from params
_SYNC_WAIT_SEC           = 0.8      # set from params

# -----------------------------
# MarkerArray callback (boxes)
# -----------------------------
def _marker_array_cb(msg: MarkerArray):
    """
    Boxes aati hi is callback se _cluster_boxes update hotay hain.
    Stamp logic:
      - default: msg.header.stamp (agar 0 ho to now)
      - agar param ~use_wall_time_for_boxes true ho to rospy.Time.now()
    """
    global _cluster_boxes, _last_boxes_stamp

    box_list = []
    for m in msg.markers:
        if m.type != Marker.CUBE:
            continue
        box_list.append({
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

    _cluster_boxes = box_list

    if _USE_WALL_TIME_FOR_BOXES:
        _last_boxes_stamp = rospy.Time.now()
    else:
        _last_boxes_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

# -----------------------------
# Small helpers
# -----------------------------
def _yaw_from_quat(qx, qy, qz, qw):
    r, p, y = euler_from_quaternion([qx, qy, qz, qw])
    return y  # radians

def _read_kitti_lines(label_file):
    """Return list of dicts for non-Don'tCare (original order)."""
    out = []
    with open(label_file) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 15:
                continue
            if parts[0].lower() == "dontcare":
                continue
            out.append({
                "type": parts[0],
                "h": float(parts[8]), "w": float(parts[9]), "l": float(parts[10]),
                "rot_y": float(parts[14]),
            })
    return out

def _load_gt_velo(label_file, calib_file):
    """
    Run kitti_label_to_velo.py to get centers in velo.
    Pair with label file to get (l,w) and rot_y; convert yaw (cam) -> yaw (lidar approx).
    Return: [(cls, np.array([x,y,z]), (l,w), yaw_lidar)]
    """
    cmd = f"rosrun my_lidar_processing kitti_label_to_velo.py {calib_file} {label_file}"
    try:
        out = subprocess.check_output(cmd, shell=True).decode("utf-8").strip().split("\n")
    except subprocess.CalledProcessError as e:
        rospy.logerr("Label conversion failed.\nCommand: %s\nOutput:\n%s",
                     cmd, e.output.decode('utf-8') if e.output else str(e))
        return []

    meta = _read_kitti_lines(label_file)  # same order assumption
    gt = []
    mi = 0
    for line in out:
        if not line.strip():
            continue
        parts = line.split()
        cls = parts[0]
        if mi >= len(meta):
            break
        coords = [float(x.strip("(),")) for x in parts[-3:]]
        l = meta[mi]["l"]; w = meta[mi]["w"]; rot_y = meta[mi]["rot_y"]
        # common heuristic: lidar yaw ≈ -(cam_rot_y + 90deg)
        yaw_lidar = -(rot_y + math.pi/2.0)
        gt.append((cls, np.array(coords, dtype=float), (l, w), yaw_lidar))
        mi += 1
    return gt

# ---------- OBB IoU (2D polygon overlap) ----------
def _rect_corners(cx, cy, sx, sy, yaw):
    hx, hy = sx * 0.5, sy * 0.5
    local = np.array([[ hx,  hy],
                      [-hx,  hy],
                      [-hx, -hy],
                      [ hx, -hy]], dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    rot = (local @ R.T)
    rot[:, 0] += cx
    rot[:, 1] += cy
    return rot  # (4,2)

def _poly_area(pts):
    x = pts[:, 0]; y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _clip_poly(subject, clipper):
    """Sutherland–Hodgman polygon clipping."""
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0.0
    def intersect(a, b, c, d):
        ba = b - a; dc = d - c
        denom = ba[0]*dc[1] - ba[1]*dc[0]
        if abs(denom) < 1e-12:
            return None
        t = ((c[0]-a[0])*dc[1] - (c[1]-a[1])*dc[0]) / denom
        return a + t*ba
    output = subject.copy()
    for i in range(len(clipper)):
        input_list = output
        output = []
        A = clipper[i]
        B = clipper[(i+1) % len(clipper)]
        if len(input_list) == 0:
            break
        S = input_list[-1]
        for E in input_list:
            if inside(E, A, B):
                if not inside(S, A, B):
                    I = intersect(S, E, A, B)
                    if I is not None:
                        output.append(I)
                output.append(E)
            elif inside(S, A, B):
                I = intersect(S, E, A, B)
                if I is not None:
                    output.append(I)
            S = E
        output = np.array(output, dtype=float)
    return output

def obb_iou_2d(ca, sa, ya, cb, sb, yb):
    A = _rect_corners(ca[0], ca[1], sa[0], sa[1], ya)
    B = _rect_corners(cb[0], cb[1], sb[0], sb[1], yb)
    inter_poly = _clip_poly(A, B)
    inter = 0.0 if inter_poly is None or len(inter_poly) == 0 else _poly_area(inter_poly)
    area_a = _poly_area(A)
    area_b = _poly_area(B)
    union = area_a + area_b - inter + 1e-12
    return float(max(0.0, inter) / union)

# ---------------------------------------------------------
# Node
# ---------------------------------------------------------
class ClusterMatcher:
    def __init__(self, calib_file, label_file):
        global _USE_WALL_TIME_FOR_BOXES, _FORCE_PROCESS_IF_SKEW, _SYNC_WAIT_SEC

        rospy.init_node("realtime_cluster_matcher", anonymous=True)

        # pubs
        self.pub_marker = rospy.Publisher("/matched_gt_marker", Marker, queue_size=10)

        # params (gates + sync)
        self.dist_max           = rospy.get_param("~match_distance_max", 6.0)
        self.iou_min            = rospy.get_param("~match_iou_min", 0.05)
        self.use_obb            = rospy.get_param("~use_obb_iou", True)
        _SYNC_WAIT_SEC          = rospy.get_param("~sync_wait_sec", 0.8)
        _USE_WALL_TIME_FOR_BOXES= rospy.get_param("~use_wall_time_for_boxes", True)
        _FORCE_PROCESS_IF_SKEW  = rospy.get_param("~force_process_if_skew", True)
        self.allow_no_boxes     = rospy.get_param("~allow_no_boxes", True)  # distance-only fallback

        # logs (overwrite per frame)
        self.kitti_id = os.path.splitext(os.path.basename(label_file))[0]
        self.log_dir = os.path.expanduser("~/det_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"{self.kitti_id}.txt")

        # GT
        self.gt_objects = _load_gt_velo(label_file, calib_file)
        if not self.gt_objects:
            rospy.logwarn("No GT objects loaded!")
        else:
            rospy.loginfo("Loaded %d GT objects:", len(self.gt_objects))
            for (t, p, lw, yaw) in self.gt_objects:
                rospy.loginfo("  %s at %s (l=%.2f, w=%.2f, yaw=%.2f°)",
                              t, p, lw[0], lw[1], math.degrees(yaw))

        # subs
        rospy.Subscriber("/pcl_centroids", PointCloud2, self.centroid_callback, queue_size=1)
        rospy.Subscriber("lidar_bounding_boxes", MarkerArray, _marker_array_cb, queue_size=1)

        rospy.loginfo("Subscribed to /pcl_centroids and lidar_bounding_boxes ... ready to match!")

    def _append_line(self, lines, det_type, cpos, box, score):
        if box is not None:
            l = max(0.1, box['sx']); w = max(0.1, box['sy']); h = max(0.1, box['sz'])
            yaw = _yaw_from_quat(*box['quat'])
        else:
            l, w, h, yaw = 1.2, 0.6, 1.7, 0.0
        lines.append(f"{det_type} {cpos[0]:.3f} {cpos[1]:.3f} {cpos[2]:.3f} "
                     f"{l:.3f} {w:.3f} {h:.3f} {yaw:.6f} {score:.3f}")

    def centroid_callback(self, msg: PointCloud2):
        global _last_centroids_stamp
        # centroids collect
        centroids = [np.array([p[0], p[1], p[2]], dtype=float)
                     for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)]
        _last_centroids_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

        # handle missing boxes
        have_boxes = (len(_cluster_boxes) > 0 and _last_boxes_stamp is not None)
        if not have_boxes and not self.allow_no_boxes:
            rospy.loginfo_throttle(2.0, "Waiting for lidar_bounding_boxes ...")
            open(self.log_path, "w").close()
            return

        # stamp skew (if boxes exist)
        if have_boxes:
            dt = abs((_last_centroids_stamp - _last_boxes_stamp).to_sec())
            rospy.loginfo_throttle(2.0,
                f"stamps: cent={_last_centroids_stamp.to_sec():.3f}, "
                f"boxes={_last_boxes_stamp.to_sec():.3f}, |Δ|={dt:.3f}s")
            if dt > _SYNC_WAIT_SEC and not _FORCE_PROCESS_IF_SKEW:
                rospy.loginfo_throttle(2.0, f"Stamp skew {dt:.3f}s > sync_wait_sec={_SYNC_WAIT_SEC:.2f}s. Waiting...")
                open(self.log_path, "w").close()
                return
            elif dt > _SYNC_WAIT_SEC and _FORCE_PROCESS_IF_SKEW:
                rospy.logwarn_throttle(2.0, f"[FORCE] processing despite skew {dt:.3f}s (> {_SYNC_WAIT_SEC:.2f}s)")

        if not centroids:
            rospy.loginfo_throttle(5, "No centroids to match.")
            open(self.log_path, "w").close()
            return

        incoming_frame_id = msg.header.frame_id if msg.header.frame_id else "velodyne"

        used_cluster = set()
        out_lines = []
        matched_any = False

        for (gt_type, gt_pos, (gl, gw), gyaw) in self.gt_objects:
            # nearest centroid (distance)
            best_c, best_d = None, float('inf')
            for cpos in centroids:
                d = np.linalg.norm(cpos - gt_pos)
                if d < best_d:
                    best_d, best_c = d, cpos

            # IoU compute (if boxes available)
            best_iou, best_idx = 0.0, -1
            if have_boxes and best_c is not None:
                for idx, b in enumerate(_cluster_boxes):
                    if idx in used_cluster:
                        continue
                    cyaw = _yaw_from_quat(*b['quat'])
                    if self.use_obb:
                        iou = obb_iou_2d(
                            ca=np.array([b['cx'], b['cy']]), sa=np.array([b['sx'], b['sy']]), ya=cyaw,
                            cb=np.array([gt_pos[0], gt_pos[1]]), sb=np.array([gl, gw]), yb=gyaw
                        )
                    else:
                        # AABB fallback
                        def aabb(cx, cy, sx, sy):
                            hx, hy = sx*0.5, sy*0.5
                            return (cx-hx, cy-hy, cx+hx, cy+hy)
                        def iou_aabb(a, b):
                            ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
                            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                            iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
                            inter = iw * ih
                            area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
                            area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
                            return inter / (area_a + area_b - inter + 1e-9)
                        iou = iou_aabb(
                            a=aabb(b['cx'], b['cy'], b['sx'], b['sy']),
                            b=aabb(gt_pos[0], gt_pos[1], gl, gw)
                        )
                    if iou > best_iou:
                        best_iou, best_idx = iou, idx

            # gating
            if have_boxes:
                is_match = (best_c is not None and best_d <= self.dist_max and best_iou >= self.iou_min)
            else:
                # distance-only fallback if no boxes present (and allowed)
                is_match = (best_c is not None and best_d <= self.dist_max)
                if is_match:
                    best_iou = 1.0  # placeholder

            if is_match:
                matched_any = True
                print(f"Matched {gt_type} (distance = {best_d:.2f} m, IoU = {best_iou:.2f})")

                # viz sphere at centroid
                m = Marker()
                m.header = Header(stamp=rospy.Time.now(), frame_id=incoming_frame_id)
                m.ns = "matched_gt"; m.id = int(np.random.randint(0, 10000))
                m.type = Marker.SPHERE; m.action = Marker.ADD
                m.pose.position.x = float(best_c[0])
                m.pose.position.y = float(best_c[1])
                m.pose.position.z = float(best_c[2])
                m.pose.orientation.w = 1.0
                m.scale.x = 0.7; m.scale.y = 0.7; m.scale.z = 0.7
                m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0
                self.pub_marker.publish(m)

                # nearest cluster box (if any) for sizes/yaw in log
                chosen_box = None
                if have_boxes:
                    min_bev = 1e9
                    for b in _cluster_boxes:
                        d_bev = (b['cx']-best_c[0])**2 + (b['cy']-best_c[1])**2
                        if d_bev < min_bev:
                            min_bev, chosen_box = d_bev, b

                self._append_line(out_lines, gt_type, best_c, chosen_box, score=float(best_iou))
                if best_idx >= 0:
                    used_cluster.add(best_idx)
            else:
                d_str = f"{best_d:.2f}" if np.isfinite(best_d) else "inf"
                print(f"No match for {gt_type} (best_d = {d_str}, best_iou = {best_iou:.2f})")

        if not matched_any:
            rospy.loginfo_throttle(3.0, "No matches passed the gates in this frame.")

        # overwrite log with latest detections only
        with open(self.log_path, "w") as f:
            if out_lines:
                f.write("\n".join(out_lines) + "\n")
            # else: empty file = no detections

# -----------------------------
# Main
# -----------------------------
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
