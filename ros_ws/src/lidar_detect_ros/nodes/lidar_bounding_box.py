#!/usr/bin/env python3
import rospy
import numpy as np
import struct
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import KDTree
import math
import random

def make_pc2_from_xyz(header, xyz_array):
    pts = np.asarray(xyz_array, dtype=np.float32)
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = pts.shape[0] if pts.size else 0
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = False
    if pts.size:
        buff = bytearray()
        for p in pts:
            buff.extend(struct.pack('fff', float(p[0]), float(p[1]), float(p[2])))
        msg.data = bytes(buff)
    else:
        msg.data = b''
    return msg

class LidarClusterNode:
    def __init__(self):
        rospy.init_node("lidar_bounding_box_node", anonymous=True)

        self.marker_pub = rospy.Publisher("lidar_bounding_boxes", MarkerArray, queue_size=1)
        self.cent_pub = rospy.Publisher("/pcl_centroids", PointCloud2, queue_size=1)

        # DBSCAN params
        self.eps = rospy.get_param("~dbscan_eps", 1.0)        # fallback eps
        self.min_samples = rospy.get_param("~dbscan_min_samples", 5)  # small-object friendly

        # Ground segmentation / polar grid params (paper-inspired defaults)
        self.region_bounds = rospy.get_param("~region_bounds", [0.5, 10.0, 30.0, 60.0, 100.0])  # 4 regions
        self.region_grid_counts = rospy.get_param("~region_grid_counts", [[2,8],[8,64],[12,128],[6,64]])
        self.alpha_h_a = rospy.get_param("~delta_h_a", 0.025)  # paper 'a' in Δh = a*x + b
        self.alpha_h_b = rospy.get_param("~delta_h_b", -0.05)  # paper 'b'
        self.seed_iter_max = rospy.get_param("~seed_iter_max", 5)
        self.ref_normal_angle_thresh = rospy.get_param("~ref_normal_angle_thresh_deg", 3.3)
        self.plane_dist_thresh = rospy.get_param("~plane_dist_thresh", 0.2)  # D
        self.k_pre_default = rospy.get_param("~k_pre_default", 5)

        # LiDAR angular resolution used for adaptive eps formula (degrees)
        self.delta_alpha = rospy.get_param("~lidar_horiz_res_deg", 0.16)  # degrees
        self.lambda_eps = rospy.get_param("~lambda_eps", 1.3)

        # (B) RANSAC fallback control (added)
        self.use_ransac_after_paper = rospy.get_param("~use_ransac_after_paper", False)

        # (C) Core search mode selector (added): "stable" or "improved"
        self.core_search_mode = rospy.get_param("~core_search_mode", "stable")

        # OBB: no params required here; can add if needed
        rospy.Subscriber("/velodyne_points", PointCloud2, self.callback, queue_size=1)
        rospy.loginfo("LIDAR bounding box node started... (eps=%s min_samples=%s)", self.eps, self.min_samples)

    # -----------------------------
    # Utilities for coordinate and grids
    # -----------------------------
    def _cart_to_polar(self, xy):
        x = xy[:,0]
        y = xy[:,1]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        return r, theta

    def _create_multi_region_grid(self, points_xy):
        r, theta = self._cart_to_polar(points_xy)
        rb = np.asarray(self.region_bounds, dtype=float)
        if rb.ndim == 0 or rb.size < 2:
            rb = np.array([0.0, 100.0])
        region_limits = []
        for i in range(len(rb)-1):
            region_limits.append((rb[i], rb[i+1]))
        region_limits = np.array(region_limits)
        num_regions = region_limits.shape[0]

        grid_to_points = {}
        rgc = list(self.region_grid_counts)
        while len(rgc) < num_regions:
            rgc.append(rgc[-1])

        for m in range(num_regions):
            Lmin_m, Lmax_m = region_limits[m]
            Nr_m, Ntheta_m = rgc[m]
            radial_edges = np.linspace(Lmin_m, Lmax_m, Nr_m+1)
            angular_edges = np.linspace(-np.pi, np.pi, Ntheta_m+1)
            in_region = (r >= Lmin_m) & (r < Lmax_m)
            if np.sum(in_region) == 0:
                for i in range(Nr_m):
                    for j in range(Ntheta_m):
                        gid = (m,i,j)
                        grid_to_points[gid] = []
                continue
            idxs = np.nonzero(in_region)[0]
            r_sub = r[idxs]
            theta_sub = theta[idxs]
            radial_bin = np.minimum(np.searchsorted(radial_edges, r_sub, side='right')-1, Nr_m-1)
            angular_bin = np.minimum(np.searchsorted(angular_edges, theta_sub, side='right')-1, Ntheta_m-1)
            for k, pt_idx in enumerate(idxs):
                i_rad = int(radial_bin[k])
                j_ang = int(angular_bin[k])
                gid = (m, i_rad, j_ang)
                if gid not in grid_to_points:
                    grid_to_points[gid] = []
                grid_to_points[gid].append(int(pt_idx))
        return grid_to_points, region_limits

    # -----------------------------
    # IMPROVED: Complete Heuristic Rules Implementation
    # -----------------------------
    def _heuristic_special_point_filter(self, raw_points):
        """EXISTING FUNCTION - UNCHANGED"""
        N = raw_points.shape[0]
        mask_candidate = np.zeros(N, dtype=bool)
        xy = raw_points[:, :2]
        r, theta = self._cart_to_polar(xy)
        theta_bins = np.round(theta / (np.deg2rad(0.5))).astype(int)
        unique_bins = np.unique(theta_bins)
        alpha_th = np.deg2rad(20.0)
        for b in unique_bins:
            idxs = np.nonzero(theta_bins == b)[0]
            if idxs.size < 2:
                continue
            idxs_sorted = idxs[np.argsort(r[idxs])]
            for k in range(1, len(idxs_sorted)):
                i_prev = idxs_sorted[k-1]
                i_cur = idxs_sorted[k]
                dz = raw_points[i_cur,2] - raw_points[i_prev,2]
                dx = raw_points[i_cur,0] - raw_points[i_prev,0]
                dy = raw_points[i_cur,1] - raw_points[i_prev,1]
                horiz = math.sqrt(dx*dx + dy*dy) + 1e-8
                ang = math.atan2(abs(dz), horiz)
                if ang > alpha_th:
                    mask_candidate[i_cur] = True
                r_cur = math.sqrt(raw_points[i_cur,0]**2 + raw_points[i_cur,1]**2)
                r_prev = math.sqrt(raw_points[i_prev,0]**2 + raw_points[i_prev,1]**2)
                if r_cur < r_prev:
                    mask_candidate[i_cur] = True
        return mask_candidate

    def _adaptive_delta_h(self, distances):
        """EXISTING FUNCTION - UNCHANGED"""
        return self.alpha_h_a * distances + self.alpha_h_b

    def _is_approach_point(self, ordered_idxs, points, delta_h_arr, i_pos):
        """NEW: Check if current point is an approach point"""
        if i_pos < 2:  # Need at least 2 previous points
            return False
            
        i_global = ordered_idxs[i_pos]
        z_cur = points[i_global, 2]
        
        # Check height differences with previous points
        z_prev1 = points[ordered_idxs[i_pos-1], 2]
        z_prev2 = points[ordered_idxs[i_pos-2], 2]
        
        dh = delta_h_arr[i_pos]
        
        # Approach condition: positive height differences exceeding threshold
        return (z_prev1 - z_cur) > dh and (z_prev2 - z_cur) > dh

    def _is_departure_point(self, ordered_idxs, points, delta_h_arr, j_pos):
        """NEW: Check if current point is a departure point"""
        if j_pos >= len(ordered_idxs) - 1:  # Need at least 1 next point
            return False
            
        j_global = ordered_idxs[j_pos]
        z_j = points[j_global, 2]
        z_next1 = points[ordered_idxs[j_pos+1], 2]
        
        dh = delta_h_arr[j_pos]
        
        # Departure condition: negative height difference below threshold
        return (z_next1 - z_j) < 0 and abs(z_next1 - z_j) < dh

    def _find_object_chain(self, ordered_idxs, points, delta_h_arr, special_mask, start_pos):
        """NEW: Find complete object chain from approach to departure point"""
        N = len(ordered_idxs)
        object_chain = []
        
        # Find approach point
        approach_pos = -1
        for i_pos in range(start_pos, min(start_pos + 10, N - 2)):  # Search window
            if not special_mask[ordered_idxs[i_pos]]:
                continue
                
            if self._is_approach_point(ordered_idxs, points, delta_h_arr, i_pos):
                approach_pos = i_pos
                object_chain.append(ordered_idxs[i_pos])
                break
        
        if approach_pos == -1:
            return [], start_pos + 1  # No approach point found
            
        # Find departure point and interior points
        departure_pos = -1
        for j_pos in range(approach_pos + 1, min(approach_pos + 20, N - 1)):  # Search window
            if not special_mask[ordered_idxs[j_pos]]:
                # Still add as interior point even if not special
                object_chain.append(ordered_idxs[j_pos])
                continue
                
            if self._is_departure_point(ordered_idxs, points, delta_h_arr, j_pos):
                departure_pos = j_pos
                object_chain.append(ordered_idxs[j_pos])
                break
            else:
                object_chain.append(ordered_idxs[j_pos])
        
        # If no departure found, use limited chain
        if departure_pos == -1:
            departure_pos = min(approach_pos + 5, N - 1)  # Limit chain length
            for k in range(approach_pos + 1, departure_pos + 1):
                if k < N:
                    object_chain.append(ordered_idxs[k])
        
        return object_chain, departure_pos + 1

    def _coarse_object_filtering(self, points):
        """IMPROVED: Now with complete object chain detection"""
        xy = points[:, :2]
        r, _ = self._cart_to_polar(xy)
        special_mask = self._heuristic_special_point_filter(points)
        N = points.shape[0]
        removed_mask = np.zeros(N, dtype=bool)
        theta = np.arctan2(points[:,1], points[:,0])
        theta_bins = np.round(theta / (np.deg2rad(0.5))).astype(int)
        unique_bins = np.unique(theta_bins)
        
        for b in unique_bins:
            idxs = np.nonzero(theta_bins == b)[0]
            if idxs.size < 4:
                continue
                
            r_sub = r[idxs]
            order = np.argsort(r_sub)
            ordered_idxs = idxs[order]
            dists = r[ordered_idxs]
            delta_h_arr = self._adaptive_delta_h(dists)
            
            # NEW: Complete object chain detection
            pos = 0
            while pos < len(ordered_idxs) - 3:
                object_chain, next_pos = self._find_object_chain(
                    ordered_idxs, points, delta_h_arr, special_mask, pos
                )
                
                # Mark all points in the chain as object points
                for idx in object_chain:
                    removed_mask[idx] = True
                
                pos = next_pos
                
                # Safety break
                if pos >= len(ordered_idxs):
                    break

        return removed_mask

    def _select_seed_points_double_threshold(self, points, grid_to_points):
        """EXISTING FUNCTION - UNCHANGED"""
        N = points.shape[0]
        is_seed = np.zeros(N, dtype=bool)
        if N == 0:
            return is_seed
        global_avg = np.mean(points[:,2])
        for gid, idx_list in grid_to_points.items():
            if len(idx_list) == 0:
                continue
            idxs = np.array(idx_list, dtype=int)
            local_avg = np.mean(points[idxs,2])
            thresh = min(local_avg, global_avg)
            is_seed[idxs[points[idxs,2] <= thresh]] = True
        for _ in range(self.seed_iter_max - 1):
            new_seed = np.zeros_like(is_seed)
            for gid, idx_list in grid_to_points.items():
                if len(idx_list) == 0:
                    continue
                idxs = np.array(idx_list, dtype=int)
                seed_idxs = idxs[is_seed[idxs]]
                if seed_idxs.size == 0:
                    local_avg = np.mean(points[idxs,2])
                else:
                    local_avg = np.mean(points[seed_idxs,2])
                thresh = min(local_avg, global_avg)
                new_seed[idxs[points[idxs,2] <= thresh]] = True
            if np.array_equal(new_seed, is_seed):
                break
            is_seed = new_seed
        return is_seed

    def _fit_plane_to_points(self, pts):
        """EXISTING FUNCTION - UNCHANGED"""
        if pts.shape[0] < 3:
            return None, None
        X = np.c_[pts[:,0], pts[:,1], np.ones(pts.shape[0])]
        y = pts[:,2]
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b, c = coef[0], coef[1], coef[2]
            normal = np.array([-a, -b, 1.0], dtype=float)
            norm = np.linalg.norm(normal) + 1e-8
            normal_unit = normal / norm
            return (a, b, c), normal_unit
        except Exception:
            return None, None

    def _multi_region_ground_segmentation(self, points):
        """EXISTING FUNCTION - UNCHANGED"""
        N = points.shape[0]
        if N == 0:
            return np.zeros(0, dtype=bool)
        coarse_removed_mask = self._coarse_object_filtering(points)
        candidate_idxs = np.nonzero(~coarse_removed_mask)[0]
        candidate_points = points[candidate_idxs, :]

        grid_to_points_rel, region_limits = self._create_multi_region_grid(candidate_points[:, :2])
        grid_to_points_global = {}
        for gid, rel_idxs in grid_to_points_rel.items():
            grid_to_points_global[gid] = [int(candidate_idxs[i]) for i in rel_idxs]

        seed_mask_global = np.zeros(N, dtype=bool)
        seed_mask_global = self._select_seed_points_double_threshold(points, grid_to_points_global)

        ground_mask = np.zeros(N, dtype=bool)
        grid_keys = list(grid_to_points_global.keys())

        def neighbors_of(gid):
            m, i_rad, j_ang = gid
            neigh = []
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    if di==0 and dj==0:
                        continue
                    ng = (m, i_rad + di, j_ang + dj)
                    if ng in grid_to_points_global:
                        neigh.append(ng)
            return neigh

        grid_plane = {}
        grid_normal = {}
        for gid in grid_keys:
            idxs = grid_to_points_global.get(gid, [])
            if len(idxs) == 0:
                continue
            seed_idxs = [i for i in idxs if seed_mask_global[i]]
            if len(seed_idxs) < 3:
                seed_idxs = idxs[:]
            if len(seed_idxs) < 3:
                continue
            pts = points[np.array(seed_idxs), :]
            plane_params, normal_unit = self._fit_plane_to_points(pts)
            if plane_params is None:
                continue
            grid_plane[gid] = plane_params
            grid_normal[gid] = normal_unit

        grid_ref_normal = {}
        for gid in grid_keys:
            neighs = neighbors_of(gid)
            nsum = np.zeros(3, dtype=float)
            count = 0
            for ng in neighs:
                n = grid_normal.get(ng, None)
                if n is not None:
                    nsum += n
                    count += 1
            if count > 0:
                ref = nsum / count
                ref_norm = np.linalg.norm(ref) + 1e-8
                grid_ref_normal[gid] = ref / ref_norm
            else:
                grid_ref_normal[gid] = grid_normal.get(gid, None)

        angle_thresh_rad = math.radians(self.ref_normal_angle_thresh)
        for gid in grid_keys:
            plane = grid_plane.get(gid, None)
            n_vec = grid_normal.get(gid, None)
            ref_n = grid_ref_normal.get(gid, None)
            if plane is None or n_vec is None or ref_n is None:
                continue
            dot = np.dot(n_vec, ref_n)
            dot = max(min(dot, 1.0), -1.0)
            ang = math.acos(dot)
            if ang > angle_thresh_rad:
                neighs = neighbors_of(gid)
                combined_idxs = []
                for ng in [gid] + neighs:
                    combined_idxs.extend(grid_to_points_global.get(ng, []))
                if len(combined_idxs) >= 3:
                    pts = points[np.array(combined_idxs), :]
                    plane_params_new, normal_new = self._fit_plane_to_points(pts)
                    if plane_params_new is not None:
                        grid_plane[gid] = plane_params_new
                        grid_normal[gid] = normal_new
                        n_vec = normal_new
            plane = grid_plane.get(gid, None)
            if plane is None:
                continue
            a,b,c = plane
            idxs = grid_to_points_global.get(gid, [])
            if len(idxs) == 0:
                continue
            pts = points[np.array(idxs), :]
            z_pred = a * pts[:,0] + b * pts[:,1] + c
            dists = np.abs(pts[:,2] - z_pred)
            for ii, idx_glob in enumerate(idxs):
                if dists[ii] < self.plane_dist_thresh:
                    ground_mask[idx_glob] = True

        ground_mask_final = ground_mask & (~coarse_removed_mask)
        return ground_mask_final

    # -----------------------------
    # Improved clustering: adaptive eps per point + improved core search + merging
    # -----------------------------
    def _adaptive_eps_for_points(self, points):
        """EXISTING FUNCTION - UNCHANGED"""
        Di = np.sqrt(points[:,0]**2 + points[:,1]**2)
        eps_vals = (self.lambda_eps * math.pi * self.delta_alpha * Di) / 180.0
        eps_vals = np.maximum(eps_vals, self.eps)
        return eps_vals

    def _improved_core_search(self, points, eps_array):
        """EXISTING FUNCTION - UNCHANGED"""
        if points.shape[0] == 0:
            return np.zeros(0, dtype=bool), []
        tree = KDTree(points[:, :2])
        neighbors = tree.query_radius(points[:, :2], r=eps_array, return_distance=False)
        N = points.shape[0]
        removed = np.zeros(N, dtype=bool)
        core_indices = []
        for idx in range(N):
            if removed[idx]:
                continue
            nbrs = neighbors[idx]
            if nbrs.shape[0] >= self.min_samples:
                core_indices.append(idx)
                for nb in nbrs:
                    removed[int(nb)] = True
        core_mask = np.zeros(N, dtype=bool)
        core_mask[core_indices] = True
        return core_mask, neighbors

    # (C) Added: stable core search without early neighbor removal
    def _stable_core_search(self, points, eps_array):
        if points.shape[0] == 0:
            return np.zeros(0, dtype=bool), []
        tree = KDTree(points[:, :2])
        neighbors = tree.query_radius(points[:, :2], r=eps_array, return_distance=False)
        N = points.shape[0]
        core_mask = np.zeros(N, dtype=bool)
        for idx in range(N):
            if neighbors[idx].shape[0] >= self.min_samples:
                core_mask[idx] = True
        return core_mask, neighbors

    def _form_and_merge_clusters(self, points, core_mask, neighbors):
        """EXISTING FUNCTION - UNCHANGED"""
        N = points.shape[0]
        labels = -1 * np.ones(N, dtype=int)
        core_idxs = np.nonzero(core_mask)[0].tolist()
        if len(core_idxs) == 0:
            return labels
        clusters = []
        for cidx in core_idxs:
            nbrs = set([int(x) for x in neighbors[cidx]])
            clusters.append(nbrs)
        merged = True
        while merged:
            merged = False
            new_clusters = []
            used = [False]*len(clusters)
            for i in range(len(clusters)):
                if used[i]:
                    continue
                base = set(clusters[i])
                used[i] = True
                for j in range(i+1, len(clusters)):
                    if used[j]:
                        continue
                    if len(base.intersection(clusters[j])) > 0:
                        base |= clusters[j]
                        used[j] = True
                        merged = True
                new_clusters.append(base)
            clusters = new_clusters
        for lab, cluster_set in enumerate(clusters):
            for idx in cluster_set:
                labels[int(idx)] = lab
        for idx in range(N):
            if labels[idx] != -1:
                continue
            for core_idx in core_idxs:
                if idx in neighbors[core_idx]:
                    cluster_label = labels[core_idx]
                    if cluster_label != -1:
                        labels[idx] = cluster_label
                        break
        return labels

    # -----------------------------
    # OBB computation (PCA-based) + quaternion helper
    # -----------------------------
    def _compute_obb_pca(self, pts):
        """EXISTING FUNCTION - UNCHANGED"""
        if pts.shape[0] == 0:
            return None, None, None
        # compute centroid
        centroid = np.mean(pts, axis=0)
        # covariance of centered points
        centered = pts - centroid
        cov = np.cov(centered, rowvar=False, bias=True)
        # eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
        # reorder to descending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]  # columns are principal axes
        # Ensure right-handed coordinate system
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:,2] *= -1.0
        # project points onto axes to get min/max extents
        proj = np.dot(centered, eigvecs)  # Nx3 coordinates in PCA frame
        mins = np.min(proj, axis=0)
        maxs = np.max(proj, axis=0)
        extents = maxs - mins  # lengths along principal axes
        # center in PCA frame then convert back to world
        center_pca = (maxs + mins) / 2.0
        center_world = centroid + np.dot(center_pca, eigvecs.T)
        # rotation matrix columns = eigvecs
        R = eigvecs
        quat = self._rotmat_to_quat(R)
        return center_world, extents, quat

    def _rotmat_to_quat(self, R):
        """EXISTING FUNCTION - UNCHANGED"""
        # Ensure R is numpy array
        R = np.array(R, dtype=float).reshape((3,3))
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                w = (R[2,1] - R[1,2]) / s
                x = 0.25 * s
                y = (R[0,1] + R[1,0]) / s
                z = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                w = (R[0,2] - R[2,0]) / s
                x = (R[0,1] + R[1,0]) / s
                y = 0.25 * s
                z = (R[1,2] + R[2,1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                w = (R[1,0] - R[0,1]) / s
                x = (R[0,2] + R[2,0]) / s
                y = (R[1,2] + R[2,1]) / s
                z = 0.25 * s
        # normalize quaternion
        q = np.array([x,y,z,w], dtype=float)
        q = q / (np.linalg.norm(q) + 1e-12)
        return q.tolist()

    # -----------------------------
    # Main callback
    # -----------------------------
    def callback(self, msg):
        """EXISTING FUNCTION - UNCHANGED"""
        pts = []
        for p in pc2.read_points(msg, field_names=("x","y","z","intensity"), skip_nans=True):
            x, y, z = p[0], p[1], p[2]
            pts.append([x, y, z])
        if len(pts) == 0:
            return
        data = np.array(pts, dtype=np.float32)

        max_range = rospy.get_param("~max_range", 100.0)
        dists = np.linalg.norm(data, axis=1)
        mask = dists < max_range
        data = data[mask]

        z_min = rospy.get_param("~z_min", -3.0)
        z_max = rospy.get_param("~z_max", 3.0)
        zmask = (data[:, 2] > z_min) & (data[:, 2] < z_max)
        data = data[zmask]

        # Ground segmentation (paper-style)
        paper_ok = True  # (B) track paper-seg status (added)
        try:
            ground_mask_paper = self._multi_region_ground_segmentation(data)
            removed_by_paper = np.sum(ground_mask_paper)
            rospy.loginfo_throttle(3, f"Paper-segmentation: ground_points_found={removed_by_paper}")
            data_nonground = data[~ground_mask_paper]
        except Exception as e:
            rospy.logwarn(f"Paper-style ground segmentation failed: {e}")
            data_nonground = data.copy()
            paper_ok = False  # (B) mark failure

        # RANSAC fallback
        try:
            # (B) run only if paper failed OR user forces it via param
            if (not paper_ok) or self.use_ransac_after_paper:
                X = data_nonground[:, :2]
                y = data_nonground[:, 2]
                if X.shape[0] >= 3:
                    ransac = RANSACRegressor(residual_threshold=0.2, max_trials=100)
                    ransac.fit(X, y)
                    inlier_mask = ransac.inlier_mask_
                    plane_pred = ransac.predict(X)
                    plane_dist = np.abs(y - plane_pred)

                    D = 0.2
                    ground_mask = (plane_dist < D) & inlier_mask

                    rospy.loginfo_throttle(3, f"RANSAC: candidates={np.sum(inlier_mask)}, removed={np.sum(ground_mask)}")
                    data = data_nonground[~ground_mask]
                else:
                    data = data_nonground
            else:
                # (B) paper succeeded and user didn't force RANSAC
                data = data_nonground
        except Exception as e:
            rospy.logwarn(f"RANSAC ground removal skipped due to error: {e}")
            data = data_nonground

        # Pre-clustering + improved DBSCAN-like clustering
        k_pre = self.k_pre_default if data.shape[0] > 100 else 1
        if k_pre > 1:
            try:
                kmeans = KMeans(n_clusters=k_pre, init='k-means++', random_state=42).fit(data)
                pre_labels = kmeans.labels_
            except Exception as e:
                rospy.logwarn(f"KMeans pre-clustering failed: {e}")
                pre_labels = np.zeros(data.shape[0], dtype=int)
        else:
            pre_labels = np.zeros(data.shape[0], dtype=int)

        final_labels = -1 * np.ones(data.shape[0], dtype=int)
        next_label = 0

        for pl in np.unique(pre_labels):
            idxs = np.where(pre_labels == pl)[0]
            sub_pts = data[idxs, :]
            if sub_pts.shape[0] < 3:
                continue

            eps_array = self._adaptive_eps_for_points(sub_pts)

            # original call (kept)
            core_mask_local, neighbors_local = self._improved_core_search(sub_pts, eps_array)
            # (C) override with stable mode if selected
            if getattr(self, 'core_search_mode', 'stable') == 'stable':
                core_mask_local, neighbors_local = self._stable_core_search(sub_pts, eps_array)

            local_labels = self._form_and_merge_clusters(sub_pts, core_mask_local, neighbors_local)

            unique_local = np.unique(local_labels)
            for ul in unique_local:
                if ul == -1:
                    continue
                mask_local = (local_labels == ul)
                final_labels[idxs[mask_local]] = next_label
                next_label += 1

            unassigned = np.where(final_labels[idxs] == -1)[0]
            if unassigned.size > 0:
                try:
                    eps_use = max(self.eps, np.mean(eps_array))
                    db = DBSCAN(eps=eps_use, min_samples=self.min_samples).fit(sub_pts[unassigned])
                    db_labels = db.labels_
                    for i_local, lbl in enumerate(db_labels):
                        if lbl == -1:
                            continue
                        if final_labels[idxs[unassigned[i_local]]] == -1:
                            final_labels[idxs[unassigned[i_local]]] = next_label
                            next_label += 1
                except Exception as e:
                    rospy.logwarn(f"Fallback DBSCAN failed in precluster {pl}: {e}")

        labels = final_labels

        # Markers + OBB (PCA-based)
        marker_array = MarkerArray()
        centroids = []
        unique_labels = set(labels)
        marker_id = 0

        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = data[labels == label]
            if cluster_points.shape[0] < 3:
                continue

            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)

            # Compute OBB via PCA
            center_world, extents, quat = self._compute_obb_pca(cluster_points)
            if center_world is None:
                # fallback to AABB if PCA fails
                x_min, y_min, z_min_v = np.min(cluster_points, axis=0)
                x_max, y_max, z_max_v = np.max(cluster_points, axis=0)
                pos = [(x_min + x_max)/2.0, (y_min + y_max)/2.0, (z_min_v + z_max_v)/2.0]
                sx = max(0.001, (x_max - x_min))
                sy = max(0.001, (y_max - y_min))
                sz = max(0.001, (z_max_v - z_min_v))
                m = Marker()
                m.header.frame_id = msg.header.frame_id if msg.header.frame_id else "velodyne"
                m.header.stamp = msg.header.stamp
                m.ns = "lidar_boxes"
                m.id = marker_id
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = float(pos[0]); m.pose.position.y = float(pos[1]); m.pose.position.z = float(pos[2])
                m.pose.orientation.w = 1.0
                m.scale.x = float(sx); m.scale.y = float(sy); m.scale.z = float(sz)
            else:
                # Use oriented box
                m = Marker()
                m.header.frame_id = msg.header.frame_id if msg.header.frame_id else "velodyne"
                m.header.stamp = msg.header.stamp
                m.ns = "lidar_boxes_obb"
                m.id = marker_id
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = float(center_world[0])
                m.pose.position.y = float(center_world[1])
                m.pose.position.z = float(center_world[2])
                # quaternion from PCA rotation
                m.pose.orientation.x = float(quat[0])
                m.pose.orientation.y = float(quat[1])
                m.pose.orientation.z = float(quat[2])
                m.pose.orientation.w = float(quat[3])
                # extents are lengths along PCA axes
                sx = float(max(0.001, extents[0]))
                sy = float(max(0.001, extents[1]))
                sz = float(max(0.001, extents[2]))
                m.scale.x = sx
                m.scale.y = sy
                m.scale.z = sz

            np.random.seed(label + 12345)
            color = np.random.rand(3)
            m.color.r = float(color[0])
            m.color.g = float(color[1])
            m.color.b = float(color[2])
            m.color.a = 0.5

            marker_array.markers.append(m)
            marker_id += 1

        self.marker_pub.publish(marker_array)

        hdr = Header()
        hdr.stamp = rospy.Time.now()
        hdr.frame_id = msg.header.frame_id if msg.header.frame_id else "velodyne"
        if len(centroids) > 0:
            pc2_cent = make_pc2_from_xyz(hdr, np.array(centroids))
            self.cent_pub.publish(pc2_cent)
        else:
            empty = PointCloud2()
            empty.header = hdr
            empty.height = 1
            empty.width = 0
            empty.fields = []
            empty.is_dense = True
            empty.data = b''
            self.cent_pub.publish(empty)


if __name__ == "__main__":
    node = LidarClusterNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

