import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np


def extract_node_positions(G, x_field: str = "x", y_field: str = "y") -> Dict[Any, Tuple[float, float]]:
    """
    Extract node positions from graph node data
    
    Args:
        G: NetworkX graph with node position data
        x_field: Field name for x coordinates
        y_field: Field name for y coordinates
        
    Returns:
        Dictionary mapping node IDs to (x, y) coordinates
    """
    return {n: (d[x_field], d[y_field]) for n, d in G.nodes(data=True) 
            if x_field in d and y_field in d}


def generate_layout(G, layout_type: str = "auto", **layout_kwargs) -> Dict[Any, Tuple[float, float]]:
    """
    Generate node positions using various NetworkX layout algorithms
    
    Args:
        G: NetworkX graph
        layout_type: Type of layout to use
        **layout_kwargs: Additional arguments for the layout algorithm
        
    Returns:
        Dictionary mapping node IDs to (x, y) coordinates
    """
    layout_functions = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "random": nx.random_layout,
        "shell": nx.shell_layout,
        "spectral": nx.spectral_layout,
        "planar": nx.planar_layout,
        "fruchterman_reingold": nx.fruchterman_reingold_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
    }
    
    if layout_type == "auto":
        # Choose layout based on graph size and connectivity
        num_nodes = len(G.nodes())
        if num_nodes <= 20:
            layout_type = "spring"
        elif num_nodes <= 100:
            layout_type = "fruchterman_reingold"
        else:
            layout_type = "kamada_kawai"
    
    if layout_type not in layout_functions:
        raise ValueError(f"Unknown layout type: {layout_type}. Available: {list(layout_functions.keys())}")
    
    return layout_functions[layout_type](G, **layout_kwargs)


def get_node_positions(G, layout_type: str = "from_data", 
                      x_field: str = "x", y_field: str = "y",
                      **layout_kwargs) -> Dict[Any, Tuple[float, float]]:
    """
    Get node positions either from graph data or by generating a layout
    
    Args:
        G: NetworkX graph
        layout_type: "from_data" to extract from node attributes, or layout algorithm name
        x_field: Field name for x coordinates (when using "from_data")
        y_field: Field name for y coordinates (when using "from_data")
        **layout_kwargs: Additional arguments for layout algorithms
        
    Returns:
        Dictionary mapping node IDs to (x, y) coordinates
    """
    if layout_type == "from_data":
        pos = extract_node_positions(G, x_field, y_field)
        if not pos:
            print(f"Warning: No position data found in fields '{x_field}', '{y_field}'. Using spring layout.")
            pos = generate_layout(G, "spring", **layout_kwargs)
        return pos
    else:
        return generate_layout(G, layout_type, **layout_kwargs)


def apply_layout_transformations(pos: Dict[Any, Tuple[float, float]], 
                                scale: float = 1.0,
                                rotation: float = 0.0,
                                offset: Tuple[float, float] = (0.0, 0.0)) -> Dict[Any, Tuple[float, float]]:
    """
    Apply transformations to node positions
    
    Args:
        pos: Original positions dictionary
        scale: Scaling factor
        rotation: Rotation angle in degrees
        offset: Translation offset (x, y)
        
    Returns:
        Transformed positions dictionary
    """
    if scale == 1.0 and rotation == 0.0 and offset == (0.0, 0.0):
        return pos
    
    # Convert to numpy arrays for easier manipulation
    positions = np.array(list(pos.values()))
    
    # Scale
    if scale != 1.0:
        positions *= scale
    
    # Rotate
    if rotation != 0.0:
        angle_rad = np.radians(rotation)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        positions = positions @ rotation_matrix.T
    
    # Translate
    if offset != (0.0, 0.0):
        positions += np.array(offset)
    
    # Convert back to dictionary
    return {node: tuple(pos) for node, pos in zip(pos.keys(), positions)}


def create_geographic_layout(G, lat_field: str = "lat", lon_field: str = "lon") -> Dict[Any, Tuple[float, float]]:
    """
    Create layout from geographic coordinates
    
    Args:
        G: NetworkX graph with geographic data
        lat_field: Field name for latitude
        lon_field: Field name for longitude
        
    Returns:
        Dictionary mapping node IDs to (x, y) coordinates
    """
    pos = {}
    for n, d in G.nodes(data=True):
        if lat_field in d and lon_field in d:
            # Use longitude as x and latitude as y
            pos[n] = (d[lon_field], d[lat_field])
    
    if not pos:
        raise ValueError(f"No geographic data found in fields '{lat_field}', '{lon_field}'")
    
    return pos


def categorize_nodes(G, capacity_field: str, provision_field: str, 
                    capacity_threshold: float = 0, 
                    provision_threshold: float = 1) -> Tuple[List, List, List]:
    """
    Categorize nodes into service, consumer, and poorly provided nodes
    
    Args:
        G: NetworkX graph
        capacity_field: Field name for capacity data
        provision_field: Field name for provision data
        capacity_threshold: Threshold for service nodes
        provision_threshold: Threshold for well-provided nodes
        
    Returns:
        Tuple of (service_nodes, consumer_nodes, not_provided_nodes)
    """
    service_nodes = [
        n for n, d in G.nodes(data=True) 
        if d.get(capacity_field, 0) > capacity_threshold
    ]
    
    consumer_nodes = [n for n in G.nodes() if n not in service_nodes]
    
    not_provided_nodes = [
        n for n, d in G.nodes(data=True) 
        if d.get(provision_field, 0) < provision_threshold
    ]
    
    return service_nodes, consumer_nodes, not_provided_nodes


def draw_service_nodes(G, pos: Dict, service_nodes: List, figsize: Tuple[int, int], 
                      ax, node_color: str = "orange", edge_color: str = "black",
                      size_multiplier: int = 20, label: str = "Service node") -> None:
    """
    Draw service nodes on the graph
    
    Args:
        G: NetworkX graph
        pos: Node positions dictionary
        service_nodes: List of service node IDs
        figsize: Figure size tuple
        ax: Matplotlib axis
        node_color: Color for service nodes
        edge_color: Edge color for service nodes
        size_multiplier: Multiplier for node size based on figure height
        label: Label for legend
    """
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=service_nodes,
        node_color=node_color,
        edgecolors=edge_color,
        node_size=figsize[1] * size_multiplier,
        ax=ax,
        label=label,
    )


def draw_consumer_nodes(G, pos: Dict, consumer_nodes: List, figsize: Tuple[int, int], 
                       ax, node_color: str = "#cccccc", edge_color: str = "gray",
                       size_multiplier: int = 10, label: str = "Consumer node") -> None:
    """
    Draw consumer nodes on the graph
    
    Args:
        G: NetworkX graph
        pos: Node positions dictionary
        consumer_nodes: List of consumer node IDs
        figsize: Figure size tuple
        ax: Matplotlib axis
        node_color: Color for consumer nodes
        edge_color: Edge color for consumer nodes
        size_multiplier: Multiplier for node size based on figure height
        label: Label for legend
    """
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=consumer_nodes,
        node_color=node_color,
        edgecolors=edge_color,
        node_size=figsize[1] * size_multiplier,
        ax=ax,
        label=label,
    )


def draw_poorly_provided_nodes(G, pos: Dict, not_provided_nodes: List, 
                              figsize: Tuple[int, int], ax, 
                              node_color: str = "#ffffff00", 
                              edge_color: str = "#e93939",
                              size_multiplier: int = 50, 
                              linewidth: int = 3,
                              label: str = "Poor node") -> None:
    """
    Draw poorly provided nodes with special highlighting
    
    Args:
        G: NetworkX graph
        pos: Node positions dictionary
        not_provided_nodes: List of poorly provided node IDs
        figsize: Figure size tuple
        ax: Matplotlib axis
        node_color: Color for poorly provided nodes (transparent by default)
        edge_color: Edge color for highlighting
        size_multiplier: Multiplier for node size based on figure height
        linewidth: Width of the highlighting border
        label: Label for legend
    """
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=not_provided_nodes,
        node_color=node_color,
        edgecolors=edge_color,
        node_size=figsize[1] * size_multiplier,
        ax=ax,
        label=label,
        linewidths=linewidth,
    )


def draw_service_flow_arrows(G, pos: Dict, figsize: Tuple[int, int], ax,
                           flow_field: str = "is_service_flow",
                           arrow_style: str = "->", 
                           arrow_color: str = "#2186eb",
                           alpha: float = 0.72,
                           line_width_divisor: int = 4) -> None:
    """
    Draw arrows for service flows between nodes
    
    Args:
        G: NetworkX graph
        pos: Node positions dictionary
        figsize: Figure size tuple
        ax: Matplotlib axis
        flow_field: Field name indicating service flow edges
        arrow_style: Matplotlib arrow style
        arrow_color: Color of the arrows
        alpha: Transparency of the arrows
        line_width_divisor: Divisor for line width based on figure height
    """
    for u, v, data in G.edges(data=True):
        if data.get(flow_field, False):
            ax.annotate(
                "",
                xy=pos[v],
                xycoords="data",
                xytext=pos[u],
                textcoords="data",
                arrowprops=dict(
                    arrowstyle=arrow_style, 
                    lw=figsize[1] / line_width_divisor, 
                    color=arrow_color, 
                    alpha=alpha
                ),
                zorder=3,
            )


def configure_plot_appearance(ax, title: str, font_size: int = 12,
                            legend_location: str = "lower right",
                            show_axis: bool = False) -> None:
    """
    Configure plot appearance, title, legend, and axis
    
    Args:
        ax: Matplotlib axis
        title: Plot title
        font_size: Font size for title and legend
        legend_location: Location of the legend
        show_axis: Whether to show axis
    """
    ax.set_title(title, fontsize=font_size)
    if not show_axis:
        ax.axis("off")
    ax.legend(loc=legend_location, fontsize=font_size)


def plot_service_flows_for_month(
    G,
    capacity_field: str = "capacity_hospital",
    provision_field: str = "provision",
    figsize: Tuple[int, int] = (12, 8),
    title: str = "",
    font_size: int = 12,
    # Layout configuration
    layout_type: str = "from_data",
    x_field: str = "x",
    y_field: str = "y",
    layout_scale: float = 1.0,
    layout_rotation: float = 0.0,
    layout_offset: Tuple[float, float] = (0.0, 0.0),
    # NEW: Layout options
    use_spring_layout: bool = False,  # Flag to use spring instead of geographic
    pos: Optional[Dict] = None,  # Pre-computed layout to reuse
    return_layout: bool = False,  # Flag to return layout for reuse
    # Node styling
    service_node_color: str = "orange",
    service_node_edge_color: str = "black",
    service_node_size_multiplier: int = 20,
    consumer_node_color: str = "#cccccc",
    consumer_node_edge_color: str = "gray",
    consumer_node_size_multiplier: int = 10,
    poor_node_color: str = "#ffffff00",
    poor_node_edge_color: str = "#e93939",
    poor_node_size_multiplier: int = 50,
    poor_node_linewidth: int = 3,
    # Arrow styling
    flow_field: str = "is_service_flow",
    arrow_style: str = "->",
    arrow_color: str = "#2186eb",
    arrow_alpha: float = 0.72,
    arrow_line_width_divisor: int = 4,
    # Plot configuration
    legend_location: str = "lower right",
    show_axis: bool = False,
    # Node labels
    service_node_label: str = "Service node",
    consumer_node_label: str = "Consumer node",
    poor_node_label: str = "Poor node",
    # Thresholds
    capacity_threshold: float = 0,
    provision_threshold: float = 1,
    # Additional layout kwargs
    **layout_kwargs
) -> Optional[Dict]:
    """
    Plot service flows for a specific month/graph state
    
    Args:
        G: NetworkX graph with node and edge data
        capacity_field: Field name for capacity data
        provision_field: Field name for provision data
        figsize: Figure size tuple
        title: Plot title
        font_size: Font size for title and legend
        layout_type: Layout algorithm ("from_data", "spring", "circular", etc.)
        x_field: Field name for x coordinates (when using "from_data")
        y_field: Field name for y coordinates (when using "from_data")
        layout_scale: Scaling factor for layout
        layout_rotation: Rotation angle in degrees
        layout_offset: Translation offset (x, y)
        use_spring_layout: If True, use spring layout regardless of layout_type
        pos: Pre-computed node positions (if provided, skips layout calculation)
        return_layout: If True, return the layout dictionary for reuse
        service_node_color: Color for service nodes
        service_node_edge_color: Edge color for service nodes
        service_node_size_multiplier: Size multiplier for service nodes
        consumer_node_color: Color for consumer nodes
        consumer_node_edge_color: Edge color for consumer nodes
        consumer_node_size_multiplier: Size multiplier for consumer nodes
        poor_node_color: Color for poorly provided nodes
        poor_node_edge_color: Edge color for poorly provided nodes
        poor_node_size_multiplier: Size multiplier for poorly provided nodes
        poor_node_linewidth: Line width for poorly provided node borders
        flow_field: Field name indicating service flow edges
        arrow_style: Matplotlib arrow style
        arrow_color: Color of service flow arrows
        arrow_alpha: Transparency of arrows
        arrow_line_width_divisor: Divisor for arrow line width
        legend_location: Location of the legend
        show_axis: Whether to show plot axis
        service_node_label: Label for service nodes in legend
        consumer_node_label: Label for consumer nodes in legend
        poor_node_label: Label for poorly provided nodes in legend
        capacity_threshold: Threshold for classifying service nodes
        provision_threshold: Threshold for classifying well-provided nodes
        **layout_kwargs: Additional arguments for layout algorithms
        
    Returns:
        Node positions dictionary if return_layout=True, None otherwise
    """
    # Get or compute node positions
    if pos is not None:
        # Use provided layout
        node_positions = pos
    elif use_spring_layout:
        # Force spring layout
        node_positions = get_node_positions(G, "spring", **layout_kwargs)
    else:
        # Use specified layout type (default behavior)
        node_positions = get_node_positions(G, layout_type, x_field, y_field, **layout_kwargs)
    
    # Apply layout transformations if needed
    node_positions = apply_layout_transformations(
        node_positions, layout_scale, layout_rotation, layout_offset
    )
    
    # Categorize nodes
    service_nodes, consumer_nodes, not_provided_nodes = categorize_nodes(
        G, capacity_field, provision_field, capacity_threshold, provision_threshold
    )
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw different types of nodes
    draw_service_nodes(
        G, node_positions, service_nodes, figsize, ax, 
        service_node_color, service_node_edge_color, 
        service_node_size_multiplier, service_node_label
    )
    
    draw_consumer_nodes(
        G, node_positions, consumer_nodes, figsize, ax,
        consumer_node_color, consumer_node_edge_color,
        consumer_node_size_multiplier, consumer_node_label
    )
    
    draw_poorly_provided_nodes(
        G, node_positions, not_provided_nodes, figsize, ax,
        poor_node_color, poor_node_edge_color,
        poor_node_size_multiplier, poor_node_linewidth,
        poor_node_label
    )
    
    # Draw service flow arrows
    draw_service_flow_arrows(
        G, node_positions, figsize, ax, flow_field, arrow_style,
        arrow_color, arrow_alpha, arrow_line_width_divisor
    )
    
    # Configure plot appearance
    configure_plot_appearance(ax, title, font_size, legend_location, show_axis)
    
    plt.tight_layout()
    plt.show()
    
    # Return layout if requested
    if return_layout:
        return node_positions
    
    return None
    """
    Plot service flows for a specific month/graph state
    
    Args:
        G: NetworkX graph with node and edge data
        capacity_field: Field name for capacity data
        provision_field: Field name for provision data
        figsize: Figure size tuple
        title: Plot title
        font_size: Font size for title and legend
        layout_type: Layout algorithm ("from_data", "spring", "circular", etc.)
        x_field: Field name for x coordinates (when using "from_data")
        y_field: Field name for y coordinates (when using "from_data")
        layout_scale: Scaling factor for layout
        layout_rotation: Rotation angle in degrees
        layout_offset: Translation offset (x, y)
        service_node_color: Color for service nodes
        service_node_edge_color: Edge color for service nodes
        service_node_size_multiplier: Size multiplier for service nodes
        consumer_node_color: Color for consumer nodes
        consumer_node_edge_color: Edge color for consumer nodes
        consumer_node_size_multiplier: Size multiplier for consumer nodes
        poor_node_color: Color for poorly provided nodes
        poor_node_edge_color: Edge color for poorly provided nodes
        poor_node_size_multiplier: Size multiplier for poorly provided nodes
        poor_node_linewidth: Line width for poorly provided node borders
        flow_field: Field name indicating service flow edges
        arrow_style: Matplotlib arrow style
        arrow_color: Color of service flow arrows
        arrow_alpha: Transparency of arrows
        arrow_line_width_divisor: Divisor for arrow line width
        legend_location: Location of the legend
        show_axis: Whether to show plot axis
        service_node_label: Label for service nodes in legend
        consumer_node_label: Label for consumer nodes in legend
        poor_node_label: Label for poorly provided nodes in legend
        capacity_threshold: Threshold for classifying service nodes
        provision_threshold: Threshold for classifying well-provided nodes
        **layout_kwargs: Additional arguments for layout algorithms
    """
    # Get node positions with flexible layout options
    pos = get_node_positions(G, layout_type, x_field, y_field, **layout_kwargs)
    
    # Apply layout transformations if needed
    pos = apply_layout_transformations(pos, layout_scale, layout_rotation, layout_offset)
    
    # Categorize nodes
    service_nodes, consumer_nodes, not_provided_nodes = categorize_nodes(
        G, capacity_field, provision_field, capacity_threshold, provision_threshold
    )
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw different types of nodes
    draw_service_nodes(
        G, pos, service_nodes, figsize, ax, 
        service_node_color, service_node_edge_color, 
        service_node_size_multiplier, service_node_label
    )
    
    draw_consumer_nodes(
        G, pos, consumer_nodes, figsize, ax,
        consumer_node_color, consumer_node_edge_color,
        consumer_node_size_multiplier, consumer_node_label
    )
    
    draw_poorly_provided_nodes(
        G, pos, not_provided_nodes, figsize, ax,
        poor_node_color, poor_node_edge_color,
        poor_node_size_multiplier, poor_node_linewidth,
        poor_node_label
    )
    
    # Draw service flow arrows
    draw_service_flow_arrows(
        G, pos, figsize, ax, flow_field, arrow_style,
        arrow_color, arrow_alpha, arrow_line_width_divisor
    )
    
    # Configure plot appearance
    configure_plot_appearance(ax, title, font_size, legend_location, show_axis)
    
    plt.tight_layout()
    plt.show()


def plot_with_geographic_layout(G, **kwargs) -> Optional[Dict]:
    """
    Convenience function to plot with geographic layout (original behavior)
    
    Args:
        G: NetworkX graph
        **kwargs: Additional arguments for plot_service_flows_for_month
        
    Returns:
        Node positions if return_layout=True
    """
    return plot_service_flows_for_month(
        G, layout_type="from_data", use_spring_layout=False, **kwargs
    )


def plot_with_spring_layout(G, **kwargs) -> Optional[Dict]:
    """
    Convenience function to plot with spring layout
    
    Args:
        G: NetworkX graph
        **kwargs: Additional arguments for plot_service_flows_for_month
        
    Returns:
        Node positions if return_layout=True
    """
    return plot_service_flows_for_month(
        G, use_spring_layout=True, **kwargs
    )


def plot_with_saved_layout(G, pos: Dict, **kwargs) -> None:
    """
    Convenience function to plot with pre-computed layout
    
    Args:
        G: NetworkX graph
        pos: Pre-computed node positions
        **kwargs: Additional arguments for plot_service_flows_for_month
    """
    plot_service_flows_for_month(G, pos=pos, return_layout=False, **kwargs)


def plot_multiple_months_service_flows(
    graphs_list: List,
    month_indices: Optional[List[int]] = None,
    title_template: str = "Service flows, month {month_idx}",
    use_consistent_layout: bool = False,
    layout_from_first: bool = True,
    **kwargs
) -> None:
    """
    Plot service flows for multiple months
    
    Args:
        graphs_list: List of NetworkX graphs for different months
        month_indices: List of month indices to plot (if None, plots all)
        title_template: Template for titles with {month_idx} placeholder
        use_consistent_layout: If True, use same layout for all months
        layout_from_first: If True, compute layout from first graph; if False, from last
        **kwargs: Additional arguments passed to plot_service_flows_for_month
    """
    if month_indices is None:
        month_indices = list(range(len(graphs_list)))
    
    saved_layout = None
    
    # Get layout from specified graph if using consistent layout
    if use_consistent_layout and month_indices:
        if layout_from_first:
            layout_graph_idx = month_indices[0]
        else:
            layout_graph_idx = month_indices[-1]
        
        if layout_graph_idx < len(graphs_list):
            saved_layout = plot_service_flows_for_month(
                graphs_list[layout_graph_idx],
                title=title_template.format(month_idx=layout_graph_idx),
                return_layout=True,
                **kwargs
            )
            # Remove the layout graph from the list to avoid double plotting
            month_indices_remaining = [idx for idx in month_indices if idx != layout_graph_idx]
        else:
            month_indices_remaining = month_indices
    else:
        month_indices_remaining = month_indices
    
    # Plot remaining months
    for month_idx in month_indices_remaining:
        if month_idx < len(graphs_list):
            title = title_template.format(month_idx=month_idx)
            
            if use_consistent_layout and saved_layout is not None:
                plot_service_flows_for_month(
                    graphs_list[month_idx],
                    title=title,
                    pos=saved_layout,
                    **kwargs
                )
            else:
                plot_service_flows_for_month(
                    graphs_list[month_idx],
                    title=title,
                    **kwargs
                )
    """
    Plot service flows for multiple months
    
    Args:
        graphs_list: List of NetworkX graphs for different months
        month_indices: List of month indices to plot (if None, plots all)
        title_template: Template for titles with {month_idx} placeholder
        **kwargs: Additional arguments passed to plot_service_flows_for_month
    """
    if month_indices is None:
        month_indices = list(range(len(graphs_list)))
    
    for month_idx in month_indices:
        if month_idx < len(graphs_list):
            title = title_template.format(month_idx=month_idx)
            plot_service_flows_for_month(
                graphs_list[month_idx],
                title=title,
                **kwargs
            )


# Example usage functions
def example_layout_options():
    """Examples of different layout options"""
    # Using data from graph (original behavior)
    # plot_service_flows_for_month(
    #     net.stats.graphs[0],
    #     layout_type="from_data",
    #     x_field="x", y_field="y"
    # )
    
    # Using spring layout
    # plot_service_flows_for_month(
    #     net.stats.graphs[0],
    #     layout_type="spring",
    #     k=2,  # spring layout parameter
    #     iterations=50
    # )
    
    # Using geographic coordinates
    # plot_service_flows_for_month(
    #     graph,
    #     layout_type="from_data",
    #     x_field="longitude", y_field="latitude"
    # )
    
    # Transformed layout
    # plot_service_flows_for_month(
    #     graph,
    #     layout_type="circular",
    #     layout_scale=2.0,
    #     layout_rotation=45,
    #     layout_offset=(100, 50)
    # )
    pass


def example_single_month_plot():
    """Example of plotting service flows for a single month"""
    # month_idx = 0
    # plot_service_flows_for_month(
    #     net.stats.graphs[month_idx],
    #     figsize=(12, 4),
    #     title=f"Service flows, month {month_idx}",
    #     capacity_field="capacity_hospital",
    #     provision_field="provision"
    # )
    pass


def example_multiple_months_plot():
    """Example of plotting service flows for multiple months"""
    # plot_multiple_months_service_flows(
    #     net.stats.graphs,
    #     month_indices=[0, 3, 6, 9],  # Plot quarterly
    #     figsize=(12, 4),
    #     capacity_field="capacity_hospital"
    # )
    pass


def example_custom_styling():
    """Example of custom styling options"""
    # plot_service_flows_for_month(
    #     net.stats.graphs[0],
    #     figsize=(15, 10),
    #     title="Custom Styled Service Flow Network",
    #     service_node_color="red",
    #     consumer_node_color="lightblue",
    #     arrow_color="green",
    #     font_size=14,
    #     poor_node_edge_color="purple",
    #     service_node_label="Hospitals",
    #     consumer_node_label="Communities",
    #     poor_node_label="Underserved Areas"
    # )
    pass


# Legacy function for backward compatibility
def plot_service_flows_for_month_legacy(G, capacity_field="capacity_hospital", 
                                       figsize=(12, 8), title="", 
                                       provision_fiels="provision"):
    """Legacy function wrapper - use plot_service_flows_for_month instead"""
    plot_service_flows_for_month(
        G=G,
        capacity_field=capacity_field,
        provision_field=provision_fiels,  # Fixed typo
        figsize=figsize,
        title=title
    )