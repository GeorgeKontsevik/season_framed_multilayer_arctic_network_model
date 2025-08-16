import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scripts.preprocesser.constants import SERVICE_COLORS, month_order

def plot_multilayer_network(
    all_results, settl_name, service_list, month=5, figsize=(15, 30)
):
    """
    Create a 3D multilayer visualization of service networks

    Parameters:
    -----------
    all_results : dict
        Dictionary containing network results
    settl_name : str
        Name of settlement to visualize
    service_list : list
        List of services to include in visualization
    month : int, optional
        Month index to visualize (default 5)
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    fig : matplotlib figure
        The created figure object
    """

    # Create the clean 3D multilayer plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")


    layer_height = 2.0  # Separation between layers
    layer_alpha = 0.05  # Transparency for layer planes

    # Get common layout for all nodes
    all_nodes = set()
    for service in service_list:
        nx_graph = all_results[settl_name][service]["graphs"][month]
        all_nodes.update(nx_graph.nodes())

    # Create master layout using first service graph
    master_graph = nx.Graph()
    master_graph.add_nodes_from(all_nodes)

    # Use the first service to get a good layout
    first_service_graph = all_results[settl_name][service_list[0]]["graphs"][month]
    pos = nx.spring_layout(first_service_graph, seed=42, k=3, iterations=100)

    # Scale positions for better 3D visualization
    for node in pos:
        pos[node] = (pos[node][0] * 6, pos[node][1] * 6)

    # Store which nodes appear on which layers
    node_layers = {node: [] for node in all_nodes}

    # Draw each service layer
    for layer_idx, service in enumerate(service_list):
        nx_graph = all_results[settl_name][service]["graphs"][month]
        z = layer_idx * layer_height
        color = SERVICE_COLORS[service]

        # Record layer membership
        for node in nx_graph.nodes():
            if node in all_nodes:
                node_layers[node].append(z)

        # Draw layer plane
        xx, yy = np.meshgrid(np.linspace(-8, 8, 20), np.linspace(-8, 8, 20))
        zz = np.ones_like(xx) * z
        ax.plot_surface(
            xx,
            yy,
            zz,
            alpha=layer_alpha,
            color=color,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        # Draw nodes
        node_x, node_y, node_z = [], [], []
        node_colors, node_sizes = [], []

        for node in nx_graph.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)

                # Check node capacity
                node_capacity = 0
                if node in all_results and service in all_results[node]:
                    node_data = all_results[node][service]
                    node_capacity = node_data.get(f"capacity_{service}", 0)

                # Set node appearance based on capacity
                if node_capacity > 0:
                    node_colors.append(color)
                    node_sizes.append(200)
                else:
                    node_colors.append("lightgray")
                    node_sizes.append(10)

        # Plot nodes
        ax.scatter(
            node_x,
            node_y,
            node_z,
            c=node_colors,
            s=node_sizes,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.0,
        )

        # Draw edges
        for edge in nx_graph.edges(data=True):
            node1, node2 = edge[0], edge[1]
            edge_data = edge[2] if len(edge) > 2 else {}

            if node1 in pos and node2 in pos:
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]

                assignment = edge_data.get("assignment", 0)

                if assignment > 0:
                    edge_color = color
                    edge_alpha = 0.9
                    edge_width = 2.5
                else:
                    edge_color = "darkgray"
                    edge_alpha = 0.9
                    edge_width = 0.3

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    [z, z],
                    color=edge_color,
                    linewidth=edge_width,
                    alpha=edge_alpha,
                )

        # Add layer label
        ax.text(
            -20,
            1,
            z - 2.3,
            service,
            fontsize=12,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=color,
                alpha=0.1,
                edgecolor="black",
                linewidth=1,
            ),
        )

    # Draw inter-layer connections
    for node, layers in node_layers.items():
        if len(layers) > 1 and node in pos:
            x, y = pos[node]
            ax.plot(
                [x] * len(layers),
                [y] * len(layers),
                layers,
                "k--",
                linewidth=0.50,
                alpha=0.3,
                linestyle="--",
            )

    # Style the plot
    ax.view_init(elev=10, azim=30)
    ax.grid(False)

    # Clean up panes
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("white")
        pane.set_alpha(0)

    # Set view limits
    ax.set_xlim(-8, 10)
    ax.set_ylim(-8, 8)

    # Add title and remove axes
    plt.title(f"Multilayer Service Network\n {month_order[month]} | {settl_name}", fontsize=16, weight="bold", y=0.93)
    ax.set_axis_off()

    plt.tight_layout()

    return fig

