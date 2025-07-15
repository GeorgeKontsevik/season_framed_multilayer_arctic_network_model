import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, MultiPolygon
import networkx as nx
from tqdm import tqdm
from constants import settl_node_color, transport_modes_color, service_node_color, FONT_SIZE

tqdm.pandas()

def plot_transport_graph(G, figsize=(14, 10), title="Transport Graph with modes", 
                        use_geo_layout=True, layout_algorithm='spring', 
                        show_labels=True, node_size=50, edge_width=1.5, 
                        alpha=0.7, save_path=None, dpi=300):
    """
    Plot a transport network graph with different transportation modes.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph with nodes containing geometry, x, y coordinates and edges with labels
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title
    use_geo_layout : bool
        If True, use geographic coordinates (x, y) from node data
        If False, use networkx layout algorithms
    layout_algorithm : str
        Layout algorithm to use when use_geo_layout=False
        Options: 'spring', 'circular', 'kamada_kawai', 'planar', 'shell'
    show_labels : bool
        Whether to show node labels
    node_size : int
        Size of nodes
    edge_width : float
        Width of edges
    alpha : float
        Transparency of edges
    save_path : str, optional
        Path to save the figure
    dpi : int
        Resolution for saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    patches = []
    
    # Draw node polygons (background) only if using geo layout
    if use_geo_layout:
        for n, d in G.nodes(data=True):
            geom = d.get("geometry")
            if isinstance(geom, Polygon) and not geom.is_empty:
                patches.append(MplPolygon(list(geom.exterior.coords), closed=True))
            elif isinstance(geom, MultiPolygon):
                for part in geom.geoms:
                    if isinstance(part, Polygon) and not part.is_empty:
                        patches.append(MplPolygon(list(part.exterior.coords), closed=True))
        
        if patches:
            pc = PatchCollection(
                patches, facecolor=settl_node_color, edgecolor="black", alpha=0.5
            )
            ax.add_collection(pc)
    
    # Determine node positions
    if use_geo_layout:
        # Use geographic coordinates
        pos = {}
        for n, d in G.nodes(data=True):
            x, y = d.get("x"), d.get("y")
            if x is not None and y is not None:
                pos[n] = (x, y)
    else:
        # Use networkx layout algorithms
        layout_functions = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'planar': nx.planar_layout,
            'shell': nx.shell_layout
        }
        
        layout_func = layout_functions.get(layout_algorithm, nx.spring_layout)
        
        # Some layouts may fail, fallback to spring layout
        try:
            if layout_algorithm == 'spring':
                pos = layout_func(G, k=1, iterations=50)
            elif layout_algorithm == 'kamada_kawai':
                pos = layout_func(G)
            elif layout_algorithm == 'planar':
                pos = layout_func(G)
            else:
                pos = layout_func(G)
        except:
            print(f"Failed to use {layout_algorithm} layout, falling back to spring layout")
            pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw edges with different colors for transport modes
    edge_counts = {}  # Count edges by transport mode
    for u, v, data in G.edges(data=True):
        label = data.get("label", "unknown")
        color = transport_modes_color.get(label, "gray")
        edge_counts[label] = edge_counts.get(label, 0) + 1
        
        if u in pos and v in pos:
            x_values = [pos[u][0], pos[v][0]]
            y_values = [pos[u][1], pos[v][1]]
            ax.plot(x_values, y_values, color=color, alpha=alpha, linewidth=edge_width)
    
    # Draw nodes
    valid_nodes = [n for n in G.nodes() if n in pos]
    valid_pos = {n: pos[n] for n in valid_nodes}
    
    if valid_nodes:
        nx.draw_networkx_nodes(
            G.subgraph(valid_nodes), valid_pos, 
            node_size=node_size, node_color=service_node_color, 
            edgecolors="black", ax=ax
        )
        
        # Node labels
        if show_labels:
            nx.draw_networkx_labels(G.subgraph(valid_nodes), valid_pos, 
                                  font_size=FONT_SIZE, ax=ax)
    
    # Legend for transport modes (only show modes that exist in the graph)
    existing_modes = set(data.get("label", "unknown") for _, _, data in G.edges(data=True))
    handles = [
        plt.Line2D([0], [0], color=transport_modes_color.get(label, "gray"), 
                  lw=2, label=f"{label} ({edge_counts.get(label, 0)})")
        for label in existing_modes if label in transport_modes_color
    ]
    
    if handles:
        ax.legend(handles=handles, title="Transport modes", 
                 loc="upper right", fontsize=FONT_SIZE)
    
    ax.set_title(title, fontsize=FONT_SIZE)
    
    if not use_geo_layout:
        ax.set_aspect("equal")
    
    ax.axis("off")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    plt.show()

# Пример использования:
# plot_transport_graph(G, use_geo_layout=True)  # С географическими координатами
# plot_transport_graph(G, use_geo_layout=False, layout_algorithm='spring')  # Spring layout
# plot_transport_graph(G, use_geo_layout=False, layout_algorithm='circular')  # Circular layout