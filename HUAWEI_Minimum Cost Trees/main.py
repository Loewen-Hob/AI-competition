import sys
import heapq


def readints():
    return list(map(int, sys.stdin.readline().split()))


def dijkstra(n, graph, start, D):
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        for v, d in graph[u]:
            if dist[v] > dist[u] + d:
                dist[v] = dist[u] + d
                heapq.heappush(heap, (dist[v], v))
    return dist


def build_tree(n, m, graph, parent, s, terminals):
    tree_edges = set()
    for t in terminals:
        path = []
        u = t
        while parent[u] != -1:
            path.append((parent[u], u))
            u = parent[u]
        for edge in path:
            tree_edges.add(edge)
    return tree_edges


def main():
    n = int(sys.stdin.readline())
    s = int(sys.stdin.readline())
    k = int(sys.stdin.readline())
    terminals = list(map(int, sys.stdin.readline().split()))
    D = int(sys.stdin.readline())
    m = int(sys.stdin.readline())
    edges = []
    graph = [[] for _ in range(n)]
    reverse_graph = [[] for _ in range(n)]
    for _ in range(m):
        a, b, c, d = map(int, sys.stdin.readline().split())
        edges.append((a, b, c, d))
        graph[a].append((b, d))
        graph[b].append((a, d))
        reverse_graph[a].append((b, d))
        reverse_graph[b].append((a, d))

    # First, compute shortest delays from s
    delay = dijkstra(n, graph, s, D)
    for t in terminals:
        if delay[t] > D:
            print(0)
            return

    # Build first tree
    dist = dijkstra(n, graph, s, D)
    parent = [-1] * n
    for u in range(n):
        for v, d in graph[u]:
            if dist[v] == dist[u] + d:
                parent[v] = u
    tree1 = build_tree(n, m, graph, parent, s, terminals)

    # Remove edges of tree1 from the graph
    remaining_graph = [[] for _ in range(n)]
    for a, b, c, d in edges:
        if (a, b) not in tree1 and (b, a) not in tree1:
            remaining_graph[a].append((b, d))
            remaining_graph[b].append((a, d))

    # Compute delay in the remaining graph
    delay2 = dijkstra(n, remaining_graph, s, D)
    for t in terminals:
        if delay2[t] > D:
            print(1)
            print(len(tree1))
            for a, b in tree1:
                print(a, b)
            return

    # Build second tree
    parent2 = [-1] * n
    for u in range(n):
        for v, d in remaining_graph[u]:
            if delay2[v] == delay2[u] + d:
                parent2[v] = u
    tree2 = build_tree(n, m, remaining_graph, parent2, s, terminals)

    print(2)
    print(len(tree1))
    for a, b in tree1:
        print(a, b)
    print(len(tree2))
    for a, b in tree2:
        print(a, b)


if __name__ == '__main__':
    main()