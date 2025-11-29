import numpy as np
import networkx as nx
from math import hypot
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

def get_closest_endpoint(graph, point):
    min_dist = float('inf')
    closest_ep = None
    for ep, _ in graph.endpoints:
        dist = (ep['o'][0] - point[0])**2 + (ep['o'][1] - point[1])**2
        if dist < min_dist:
            min_dist = dist
            closest_ep = ep
    return closest_ep

def compute_edge_radius(edge, edt): 
    y, x = zip(*edge['pts'])
    return np.mean(edt[y,x])

def _copy_graph(G: nx.Graph):
    """Deep-copy a networkx graph with all node/edge attributes."""
    H = nx.Graph()
    for n, data in G.nodes(data=True):
        H.add_node(n, **data.copy())
    for u, v, data in G.edges(data=True):
        H.add_edge(u, v, **data.copy())
    return H

def add_edge_to_skeleton(skel, edge):
    y, x = zip(*edge['pts'])
    skel[y,x] = 1
    return skel

def explore_graph(large_skel, full_graph, root_node):
    neighbors = list(full_graph.G.neighbors(root_node))
    if len(neighbors) > 0:
        max_radius=float('-inf')
        next_node = None
        for n in neighbors:
            if full_graph.G[root_node][n]['radius'] >= max_radius:
                max_radius = full_graph.G[root_node][n]['radius']
                next_node = n
        
        large_skel = add_edge_to_skeleton(large_skel, full_graph.G[root_node][next_node])
        full_graph.G.remove_node(root_node)
        
        return explore_graph(large_skel, full_graph, next_node)
    return large_skel, full_graph

class SkeletonGraph:
    """
    Wrapper for the user-provided graph format.
    Attributes:
        G: networkx Graph where:
            - graph.edges[s,e]['pts'] is an array of (y,x) points
            - graph.nodes[n]['o'] is the node coordinate (y,x)
        endpoints: List of nodes with degree 1
        intersections: List of nodes with degree >=3
    """
    def __init__(self, graph, edt=None):
        self.G = graph
        for (u,v) in self.G.edges():
            if u == v:
                self.G.remove_edge(u,v)
            elif edt is not None:
                self.G[u][v]['radius'] = compute_edge_radius(self.G[u][v], edt)

        self.endpoints = [(self.G.nodes[n], n) for n in self.G.nodes() if self.G.degree[n] == 1]
        self.intersections = [(self.G.nodes[n], n) for n in self.G.nodes() if self.G.degree[n] > 2]

    def nodes(self):
        return self.G.nodes

    def edges(self):
        out = []
        for u, v in self.G.edges():
            pts = self.G[u][v]['pts']
            out.append((u, v, pts))
        return out

    def to_mask(self, shape):
        M = np.zeros(shape, dtype=np.uint8)
        for _, _, pts in self.edges():
            pts_rounded = np.round(pts).astype(int)
            M[pts_rounded[:,0], pts_rounded[:,1]] = 1
        return M

    def display(self):
        # draw edges by pts
        for (s,e) in self.G.edges():
            ps = self.G[s][e]['pts']
            plt.plot(ps[:,1], ps[:,0], 'green')

        # draw node by o
        nodes = self.G.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:,1], ps[:,0], 'r.')

        # title and show
        plt.title('Build Graph')
        plt.show()

    def get_node(self, x, y):
        for i in self.G.nodes():
            oy, ox = self.G.nodes[i]['o']
            if int(ox) == int(x) and int(oy) == int(y):
                return self.G.nodes[i], i
        return None
    
    def get_closest_node(self, point):
        min_dist = float('inf')
        closest_ep = None
        for n in self.G.nodes():
            node = self.G.nodes[n]
            dist = (node['o'][0] - point[0])**2 + (node['o'][1] - point[1])**2
            if dist < min_dist:
                min_dist = dist
                closest_ep = node
        return closest_ep

    def add_node(self, pts, o):
        node_nb = len(self.G.nodes)
        self.G.add_node(node_nb)
        self.G.nodes[node_nb]['pts']= pts
        self.G.nodes[node_nb]['o']= o
        return node_nb

    def add_edge(self, u, v, edge):
        self.G.add_edge(u,v)
        self.G[u][v] = edge

    def remove_node(self, node):
        self.G.remove_node(node)

    def __add__(self, other):
        """
        Combine two SkeletonGraph objects.
        Nodes with identical coordinates ('o') are merged.
        All edges are preserved.
        """

        # Work on copies
        G1 = _copy_graph(self.G)
        G2 = _copy_graph(other.G)

        # Build new empty merged graph
        Gm = nx.Graph()

        # Mapping: (y,x) -> new node index
        coord_to_new = {}
        next_id = 0

        def add_from_graph(G_src):
            nonlocal next_id
            # 1) add nodes
            for n, data in G_src.nodes(data=True):
                coord = tuple(data['o'])
                if coord not in coord_to_new:
                    coord_to_new[coord] = next_id
                    Gm.add_node(next_id, **data.copy())
                    next_id += 1

            # 2) add edges with remapped endpoints
            for u, v, edata in G_src.edges(data=True):
                cu = coord_to_new[tuple(G_src.nodes[u]['o'])]
                cv = coord_to_new[tuple(G_src.nodes[v]['o'])]
                if not Gm.has_edge(cu, cv):
                    Gm.add_edge(cu, cv, **edata.copy())
                else:
                    # If already exists, do nothing or merge attributes if needed
                    pass

        add_from_graph(G1)
        add_from_graph(G2)

        return SkeletonGraph(Gm)

    def __sub__(self, other):
        """
        Remove from self all edges present in other (same geometry).
        Nodes left isolated are removed.
        """

        G1 = _copy_graph(self.G)
        G2 = other.G

        # Build a fast lookup: frozenset of pixel coordinates
        def edge_signature(pts):
            pts = np.round(pts).astype(int)
            return frozenset((int(y), int(x)) for (y, x) in pts)

        signatures_to_remove = set()
        for u, v, data in G2.edges(data=True):
            signatures_to_remove.add(edge_signature(data['pts']))

        # Remove edges in G1 whose signature matches
        to_delete = []
        for u, v, data in G1.edges(data=True):
            sig = edge_signature(data['pts'])
            if sig in signatures_to_remove:
                to_delete.append((u, v))

        for (u, v) in to_delete:
            if G1.has_edge(u, v):
                G1.remove_edge(u, v)

        # Remove isolated nodes
        dead_nodes = [n for n in G1.nodes() if G1.degree[n] == 0]
        for n in dead_nodes:
            G1.remove_node(n)

        return SkeletonGraph(G1)