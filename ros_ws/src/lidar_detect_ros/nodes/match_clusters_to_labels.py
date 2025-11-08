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
_YAW_OFFSET = 0.0
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

# --------- OBB utils + Oriented IoU (BEV) + Hungarian ---------
def obb2d_corners(cx, cy, sx, sy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = sx*0.5, sy*0.5
    pts = np.array([[ ex,  ey],
                    [ ex, -ey],
                    [-ex, -ey],
                    [-ex,  ey]], dtype=float)
    R = np.array([[c,-s],[s, c]], dtype=float)
    rot = pts @ R.T
    rot[:,0] += cx; rot[:,1] += cy
    return rot

def _poly_area(poly):
    x = poly[:,0]; y = poly[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

def _clip_polygon(subject, clip):
    def inside(a, b, p):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0.0
    def intersect(a1, a2, b1, b2):
        A = np.array([[a2[0]-a1[0], b1[0]-b2[0]],
                      [a2[1]-a1[1], b1[1]-b2[1]]], dtype=float)
        b = np.array([b1[0]-a1[0], b1[1]-a1[1]], dtype=float)
        det = A[0,0]*A[1,1]-A[0,1]*A[1,0]
        if abs(det) < 1e-12:
            return (a1 + a2)/2.0
        t = ( b[0]*A[1,1]-b[1]*A[0,1]) / det
        return a1 + t*(a2-a1)

    out = np.asarray(subject, dtype=float).copy()
    for i in range(len(clip)):
        inp = np.asarray(out, dtype=float).copy()
        out = []
        if inp.size == 0:
            break
        A = clip[i]; B = clip[(i+1) % len(clip)]
        S = inp[-1]
        for E in inp:
            if inside(A, B, E):
                if not inside(A, B, S):
                    out.append(intersect(S, E, A, B))
                out.append(E)
            elif inside(A, B, S):
                out.append(intersect(S, E, A, B))
            S = E
        out = np.asarray(out, dtype=float)
    return np.asarray(out, dtype=float)

def oriented_iou_bev(obb1, obb2):
    c1 = obb2d_corners(*obb1)   # (cx,cy,sx,sy,yaw)
    c2 = obb2d_corners(*obb2)
    inter = _clip_polygon(c1, c2)
    inter = np.asarray(inter, dtype=float)
    if inter.size < 6:  # < 3 points => area 0
        inter_area = 0.0
    else:
        inter_area = _poly_area(inter)
    a1 = _poly_area(c1); a2 = _poly_area(c2)
    union = a1 + a2 - inter_area + 1e-12
    return float(inter_area / union)

def cluster_marker_to_obb2d(box_dict):
    cx, cy = float(box_dict['cx']), float(box_dict['cy'])
    sx, sy = max(0.05, float(box_dict['sx'])), max(0.05, float(box_dict['sy']))
    qx, qy, qz, qw = box_dict['quat']
    yaw = _yaw_from_quat(qx, qy, qz, qw)
    return (cx, cy, sx, sy, yaw)

# cam_yaw_to_lidar_yaw(...) ko replace karo:
def cam_yaw_to_lidar_yaw(rot_y_cam):
    return float(-(rot_y_cam + math.pi/2.0) + _YAW_OFFSET)

def range_adaptive_gates(r):
    # near/mid/far ke liye dist aur min IoU thora relax
    if r < 15.0:  return 12.0, 0.02
    if r < 35.0:  return 15.0, 0.015
    return 18.0, 0.01

def _fused_cost(gt_obb, cl_obb):
    gcx,gcy,gl,gw,gyaw = gt_obb
    ccx,ccy,cl, cw, cyaw = cl_obb
    dist = math.hypot(gcx-ccx, gcy-ccy)
    iou  = oriented_iou_bev(gt_obb, cl_obb)
    return 0.8*dist + 0.2*(1.0 - iou), dist, iou

# module-level globals (defaults)
_MIN_IOU_FLOOR = 0.05
_MAX_DIST_CEIL = 20.0

def _build_cost_matrix(gt_list, cl_list):
    M, N = len(gt_list), len(cl_list)
    C  = np.zeros((M,N), dtype=float)
    D  = np.zeros((M,N), dtype=float)
    IOU= np.zeros((M,N), dtype=float)
    for i, g in enumerate(gt_list):
        r = math.hypot(g[0], g[1])
        max_d, min_iou = range_adaptive_gates(r)
        gate_d   = min(max_d, _MAX_DIST_CEIL)
        gate_iou = max(min_iou, _MIN_IOU_FLOOR)
        for j, c in enumerate(cl_list):
            cost, dist, iou = _fused_cost(g, c)
            C[i, j]   = cost if (dist <= gate_d and iou >= gate_iou) else 1e6
            D[i, j]   = dist
            IOU[i, j] = iou
    return C, D, IOU

def _hungarian_assign(C):
    try:
        from scipy.optimize import linear_sum_assignment
        r_idx, c_idx = linear_sum_assignment(C)
        return list(zip(r_idx.tolist(), c_idx.tolist()))
    except Exception:
        M,N = C.shape
        used_r, used_c, pairs = set(), set(), []
        flat = [(C[i,j], i, j) for i in range(M) for j in range(N)]
        for _, i, j in sorted(flat):
            if i in used_r or j in used_c: continue
            if C[i,j] >= 1e6: continue
            used_r.add(i); used_c.add(j); pairs.append((i,j))
        return pairs

def match_gt_to_clusters(gt_obbs, cl_obbs):
    if len(gt_obbs)==0 or len(cl_obbs)==0:
        return [], list(range(len(gt_obbs))), list(range(len(cl_obbs))), None, None, None
    C, D, IOU = _build_cost_matrix(gt_obbs, cl_obbs)
    pairs = _hungarian_assign(C)
    matches, G, Cc = [], set(), set()
    for gi, cj in pairs:
        if C[gi, cj] >= 1e6: continue
        matches.append({"gt_idx": gi, "cl_idx": cj,
                        "cost": float(C[gi,cj]),
                        "dist": float(D[gi,cj]),
                        "iou":  float(IOU[gi,cj])})
        G.add(gi); Cc.add(cj)
    un_gt = [i for i in range(len(gt_obbs)) if i not in G]
    un_cl = [j for j in range(len(cl_obbs)) if j not in Cc]
    return matches, un_gt, un_cl, C, D, IOU

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
    """
    Converter se velo coords (cx,cy,cz) aate hain; label se (h,w,l, rot_y_cam).
    Yahan rot_y ko LiDAR yaw mein map karke BEV OBB ready return hota hai.
    """
    cmd = f"rosrun my_lidar_processing kitti_label_to_velo.py {calib_file} {label_file}"
    try:
        out = subprocess.check_output(cmd, shell=True).decode("utf-8").strip().split("\n")
    except subprocess.CalledProcessError as e:
        rospy.logerr("Label conversion failed.\nCommand: %s\nOutput:\n%s",
                     cmd, e.output.decode("utf-8") if e.output else str(e))
        return []

    dims_list = _read_dims_from_kitti(label_file)  # (typ,h,w,l,rot_y_cam)
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
            cx, cy, cz = [float(x.strip("(),")) for x in parts[-3:]]

            if k < len(dims_list):
                typ, h, w, l, rot_y_cam = dims_list[k]; k += 1
                yaw_lidar = cam_yaw_to_lidar_yaw(rot_y_cam)
                type_name = typ
            else:
                # fallback dims + yaw
                w, l, yaw_lidar = 0.6, 1.2, 0.0
                type_name = obj_type

            gt.append({
                "type": type_name,
                "cx": float(cx), "cy": float(cy), "cz": float(cz),
                "l": float(l),   "w":  float(w),
                "yaw": float(yaw_lidar)
            })
        except Exception:
            continue
    return gt


# ---------------------------------------------------------
# Node: Match + LOG (per-frame overwrite)
# ---------------------------------------------------------
class ClusterMatcher:
    def __init__(self, calib_file, label_file):
        rospy.init_node("realtime_cluster_matcher", anonymous=True)

        global _YAW_OFFSET, _MIN_IOU_FLOOR, _MAX_DIST_CEIL
        _YAW_OFFSET    = float(rospy.get_param("~yaw_offset_rad", 0.0))
        _MIN_IOU_FLOOR = float(rospy.get_param("~min_iou_floor", 0.01))
        _MAX_DIST_CEIL = float(rospy.get_param("~max_dist_ceiling", 15.0))

        self.pub_marker = rospy.Publisher("/matched_gt_marker", Marker, queue_size=10)

        # gates
        self.dist_max = float(rospy.get_param("~match_distance_max", 6.0))
        self.iou_min  = float(rospy.get_param("~match_iou_min", 0.05))
        self.wait_window = float(rospy.get_param("~sync_wait_sec", 1.2))
        self.force_if_skew = bool(rospy.get_param("~force_if_skew", True))
        self.min_iou_floor = float(rospy.get_param("~min_iou_floor", 0.01))
        self.max_dist_ceiling = float(rospy.get_param("~max_dist_ceiling", 15.0))

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
            for g in self.gt_objects:
                rospy.loginfo(
            "  %s at (%.2f, %.2f, %.2f) (l=%.2f, w=%.2f, yaw=%.2f)",
            g["type"], g["cx"], g["cy"], g["cz"], g["l"], g["w"], g["yaw"]
        )

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
            rospy.loginfo_throttle(5, "No centroids msg; proceeding with boxes/GT only.")

        incoming_frame_id = "velodyne"
        if hasattr(msg, "header") and msg.header.frame_id:
            incoming_frame_id = msg.header.frame_id

        # --- Build OBBs (clusters & GT) and run Hungarian matching ---
        cl_obbs = [cluster_marker_to_obb2d(b) for b in _cluster_boxes]
        gt_obbs = [(g["cx"], g["cy"], g["l"], g["w"], g["yaw"]) for g in self.gt_objects]

        matches, un_gt, un_cl, Cmat, Dmat, IOUmat = match_gt_to_clusters(gt_obbs, cl_obbs)

        out_lines = []
        matched_any = len(matches) > 0

        # viz + log lines
        for mrec in matches:
            gi, cj = mrec["gt_idx"], mrec["cl_idx"]
            g = self.gt_objects[gi]
            # centroid choose = nearest point to GT in this frame (optional: use box center)
            best_c = np.array([g["cx"], g["cy"], g["cz"]], dtype=float)

            # viz marker at GT center
            m = Marker()
            m.header = Header(stamp=rospy.Time.now(), frame_id=incoming_frame_id)
            m.ns = "matched_gt"; m.id = int(np.random.randint(0, 10000))
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = float(best_c[0]); m.pose.position.y = float(best_c[1]); m.pose.position.z = float(best_c[2])
            m.pose.orientation.w = 1.0
            m.scale.x = 0.7; m.scale.y = 0.7; m.scale.z = 0.7
            m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0
            self.pub_marker.publish(m)

            chosen_box = _cluster_boxes[cj] if 0 <= cj < len(_cluster_boxes) else None
            self._append_line(out_lines, g["type"], best_c, chosen_box, score=float(mrec["iou"]))

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
