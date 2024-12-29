import sys
import math
import networkx as nx
from collections import defaultdict
from itertools import combinations
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import ConvexHull


def read_input():
    input_lines = sys.stdin.read().splitlines()
    idx = 0
    n_nodes = int(input_lines[idx]);
    idx += 1
    nodes = {}
    for _ in range(n_nodes):
        parts = input_lines[idx].split();
        idx += 1
        node_id = int(parts[0])
        lat = float(parts[1])
        lon = float(parts[2])
        nodes[node_id] = (lat, lon)
    n_edges = int(input_lines[idx]);
    idx += 1
    edges = []
    for _ in range(n_edges):
        u, v = map(int, input_lines[idx].split());
        idx += 1
        edges.append((u, v))
    n_users = int(input_lines[idx]);
    idx += 1
    users = {}
    user_coords = []
    for _ in range(n_users):
        parts = input_lines[idx].split();
        idx += 1
        user_id = int(parts[0])
        lat = float(parts[1])
        lon = float(parts[2])
        users[user_id] = (lat, lon)
        user_coords.append([lat, lon])
    return nodes, edges, users, np.array(user_coords)


def build_graph(nodes, edges):
    G = nx.Graph()
    for node_id, coord in nodes.items():
        G.add_node(node_id, coord=coord)
    for u, v in edges:
        G.add_edge(u, v)
    return G


def cluster_users(user_coords, min_size=512, max_size=1024):
    n_users = len(user_coords)
    k = math.ceil(n_users / max_size)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(user_coords)
    labels = kmeans.labels_
    clusters = defaultdict(list)
    for user_id, label in zip(range(1, n_users + 1), labels):
        clusters[label].append(user_id)
    # 调整簇以确保每个簇至少有min_size个用户
    final_clusters = []
    current = []
    for cluster in clusters.values():
        current += cluster
        if len(current) >= min_size:
            final_clusters.append(current)
            current = []
    if current:
        if len(final_clusters) == 0:
            final_clusters.append(current)
        else:
            final_clusters[-1].extend(current)
    return final_clusters


def get_convex_hull(points):
    if len(points) < 3:
        return list(range(len(points)))
    hull = ConvexHull(points)
    return hull.vertices


def haversine_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # 地球半径，单位公里
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def map_cluster_to_polygon(G, nodes, cluster_user_ids, users):
    # 获取簇中用户的坐标
    cluster_users = [users[uid] for uid in cluster_user_ids]
    points = np.array(cluster_users)
    # 计算凸包
    hull_indices = get_convex_hull(points)
    hull_points = points[hull_indices]
    # 找到最接近凸包点的路网节点
    polygon_nodes = []
    for point in hull_points:
        min_dist = float('inf')
        nearest_node = None
        for node_id, coord in nodes.items():
            dist = haversine_distance(point, coord)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id
        polygon_nodes.append(nearest_node)
    # 移除重复节点，确保多边形是简单循环
    polygon_nodes = list(dict.fromkeys(polygon_nodes))
    return polygon_nodes


def calculate_absolute_curvature(polygon, nodes):
    # 将节点坐标转换为3D空间点
    def to_3d(lat, lon):
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x = math.cos(lat_rad) * math.cos(lon_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad)
        z = math.sin(lat_rad)
        return np.array([x, y, z])

    points_3d = [to_3d(*nodes[node_id]) for node_id in polygon]
    n = len(points_3d)
    total_angle = 0.0
    for i in range(n):
        A = points_3d[i - 1]
        B = points_3d[i]
        C = points_3d[(i + 1) % n]
        # 计算向量
        BA = A - B
        BC = C - B
        # 计算夹角
        dot_product = np.dot(BA, BC)
        norm_product = np.linalg.norm(BA) * np.linalg.norm(BC)
        if norm_product == 0:
            angle = 0
        else:
            angle = math.acos(max(min(dot_product / norm_product, 1.0), -1.0))
        total_angle += abs(angle)
    AC = total_angle / (2 * math.pi)
    return AC


def main():
    nodes, edges, users, user_coords = read_input()
    G = build_graph(nodes, edges)
    clusters = cluster_users(user_coords)
    polygons = []
    polygon_user_ids = []
    for cluster in clusters:
        polygon = map_cluster_to_polygon(G, nodes, cluster, users)
        if len(polygon) < 3:
            # 如果多边形节点少于3个，无法形成闭环，需调整
            continue
        polygons.append(polygon)
        polygon_user_ids.append(cluster)
    # 计算每个多边形的绝对曲率
    ac_values = []
    for poly in polygons:
        ac = calculate_absolute_curvature(poly, nodes)
        ac_values.append(ac)
    # 找到最大绝对曲率
    max_ac = max(ac_values) if ac_values else 0
    # 输出
    print(len(polygons))
    for poly, user_ids in zip(polygons, polygon_user_ids):
        print(len(poly))
        print(' '.join(map(str, poly)))
        print(len(user_ids))
        print(' '.join(map(str, user_ids)))


if __name__ == "__main__":
    main()
