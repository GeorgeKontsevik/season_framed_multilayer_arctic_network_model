import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scicolor  # Assuming this is the scicolor import
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict


def extract_unique_node_names(results: List[pd.DataFrame], 
                             name_field: str = "name") -> List[str]:
    """
    Extract unique node names from all timesteps
    
    Args:
        results: List of DataFrames with provision data
        name_field: Name of the node name column
        
    Returns:
        Sorted list of unique node names
    """
    node_names = {
        row[name_field]
        for timestep in results
        for _, row in timestep.reset_index().iterrows()
    }
    return sorted(node_names)


def organize_data_by_month(results: List[pd.DataFrame], 
                          node_names: List[str],
                          name_field: str = "name",
                          provision_field: str = "provision") -> Dict[int, Dict[str, List[float]]]:
    """
    Organize provision data by month and node
    
    Args:
        results: List of DataFrames with provision data
        node_names: List of all node names
        name_field: Name of the node name column
        provision_field: Name of the provision column
        
    Returns:
        Dictionary with monthly data organized by node
    """
    monthly_data = {month: {name: [] for name in node_names} for month in range(12)}
    
    for i, timestep in enumerate(results):
        month = i % 12
        for _, row in timestep.reset_index().iterrows():
            monthly_data[month][row[name_field]].append(row[provision_field])
    
    return monthly_data


def create_provision_matrix(monthly_data: Dict[str, List[float]], 
                           node_names: List[str]) -> np.ndarray:
    """
    Create provision matrix for visualization
    
    Args:
        monthly_data: Dictionary with provision data by node
        node_names: Ordered list of node names
        
    Returns:
        2D numpy array with provision values (nodes x years)
    """
    return np.array([monthly_data[node] for node in node_names])


def plot_monthly_provision_subplot(ax, provision_matrix: np.ndarray, 
                                  node_names: List[str], years: List[int],
                                  month_name: str, month_index: int,
                                  colormap: str = "Hokusai2",
                                  vmin: float = 0, vmax: float = 1,
                                  show_node_labels: bool = True,
                                  show_year_labels: bool = True,
                                  grid_alpha: float = 0.2) -> Any:
    """
    Plot provision data for a single month
    
    Args:
        ax: Matplotlib axis
        provision_matrix: 2D array with provision data
        node_names: List of node names for y-axis
        years: List of years for x-axis
        month_name: Name of the month for title
        month_index: Index of the month (0-11)
        colormap: Colormap name
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        show_node_labels: Whether to show node names on y-axis
        show_year_labels: Whether to show year labels on x-axis
        grid_alpha: Transparency of grid lines
        
    Returns:
        Image object for colorbar creation
    """
    # Get colormap
    try:
        cmap = scicolor.get_cmap(colormap)
    except AttributeError:
        cmap = colormap
    
    # Create image
    im = ax.imshow(
        provision_matrix,
        aspect="auto",
        cmap=cmap,
        extent=[years[0] - 0.5, years[-1] + 0.5, -0.5, len(node_names) - 0.5],
        vmin=vmin,
        vmax=vmax,
    )
    
    # Set title
    ax.set_title(month_name)
    
    # Configure y-axis (node names)
    ax.set_yticks(range(len(node_names)))
    if show_node_labels:
        ax.set_yticklabels(node_names)
    else:
        ax.set_yticklabels([])
    
    # Configure x-axis (years)
    if show_year_labels:
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.set_xticklabels([])
    
    # Add grid
    ax.grid(True, which="major", color="w", linestyle="--", linewidth=0.2, alpha=grid_alpha)
    
    return im


def configure_subplot_grid(axes, month_index: int, total_months: int = 12,
                         grid_cols: int = 4) -> Tuple[bool, bool]:
    """
    Determine which labels to show based on subplot position
    
    Args:
        axes: Matplotlib axes array
        month_index: Current month index (0-11)
        total_months: Total number of months
        grid_cols: Number of columns in grid
        
    Returns:
        Tuple of (show_node_labels, show_year_labels)
    """
    # Show node labels only on leftmost column
    show_node_labels = (month_index % grid_cols == 0)
    
    # Show year labels only on bottom row
    rows_needed = (total_months + grid_cols - 1) // grid_cols
    bottom_row_start = (rows_needed - 1) * grid_cols
    show_year_labels = (month_index >= bottom_row_start)
    
    return show_node_labels, show_year_labels


def add_colorbar(fig, im, label: str = "Provision Value",
                position: List[float] = [0.92, 0.15, 0.015, 0.7]) -> None:
    """
    Add colorbar to the figure
    
    Args:
        fig: Matplotlib figure
        im: Image object from imshow
        label: Label for colorbar
        position: Position and size [left, bottom, width, height]
    """
    cbar_ax = fig.add_axes(position)
    fig.colorbar(im, cax=cbar_ax, label=label)


def plot_provision_evolution_by_node(
    results: List[pd.DataFrame],
    start_year: int = 2025,
    figsize: Tuple[int, int] = (30, 18),
    name_field: str = "name",
    provision_field: str = "provision",
    colormap: str = "Hokusai2",
    vmin: float = 0,
    vmax: float = 1,
    month_names: Optional[List[str]] = None,
    grid_shape: Tuple[int, int] = (3, 4),
    title: str = "Provision Evolution by Node and Month",
    font_size: Optional[int] = None,
    colorbar_label: str = "Provision Value",
    colorbar_position: List[float] = [0.92, 0.15, 0.015, 0.7],
    grid_alpha: float = 0.2,
    tight_layout_rect: List[float] = [0, 0, 0.91, 1]
) -> None:
    """
    Plot provision changes over time for each node, separated by month
    
    Args:
        results: List of DataFrames (one per timestep) with provision data
        start_year: Year of first timestep
        figsize: Size of entire plot grid
        name_field: Name of the node name column
        provision_field: Name of the provision column
        colormap: Colormap name
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        month_names: Custom month names (default: full month names)
        grid_shape: Shape of subplot grid (rows, cols)
        title: Main title for the entire figure
        font_size: Font size for title
        colorbar_label: Label for the colorbar
        colorbar_position: Position of colorbar [left, bottom, width, height]
        grid_alpha: Transparency of grid lines
        tight_layout_rect: Rectangle for tight layout
    """
    # Set default month names
    if month_names is None:
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
    
    # Extract node names
    node_names = extract_unique_node_names(results, name_field)
    
    # Organize data by month
    monthly_data = organize_data_by_month(results, node_names, name_field, provision_field)
    
    # Create subplot grid
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)
    
    # Track image for colorbar
    im = None
    
    for month in range(12):
        # Get subplot position
        ax = axes[month // grid_shape[1], month % grid_shape[1]]
        
        # Get data for this month
        data = monthly_data[month]
        
        # Check if we have data
        n_years = len(next(iter(data.values()), []))
        if n_years == 0:
            ax.set_visible(False)
            continue
        
        # Create years list
        years = list(range(start_year, start_year + n_years))
        
        # Create provision matrix
        provision_matrix = create_provision_matrix(data, node_names)
        
        # Determine label visibility
        show_node_labels, show_year_labels = configure_subplot_grid(
            axes, month, 12, grid_shape[1]
        )
        
        # Plot this month's data
        im = plot_monthly_provision_subplot(
            ax, provision_matrix, node_names, years, month_names[month], month,
            colormap, vmin, vmax, show_node_labels, show_year_labels, grid_alpha
        )
    
    # Add colorbar if we have an image
    if im is not None:
        add_colorbar(fig, im, colorbar_label, colorbar_position)
    
    # Set main title
    if font_size is None:
        font_size = max(12, figsize[0] // 3)  # Adaptive font size
    
    fig.suptitle(title, fontsize=font_size, y=1)
    
    # Apply tight layout
    plt.tight_layout(rect=tight_layout_rect)
    plt.show()


def plot_provision_evolution_subset(
    results: List[pd.DataFrame],
    selected_nodes: List[str],
    start_year: int = 2025,
    figsize: Tuple[int, int] = (30, 18),
    **kwargs
) -> None:
    """
    Plot provision evolution for a subset of nodes
    
    Args:
        results: List of DataFrames with provision data
        selected_nodes: List of node names to include
        start_year: Year of first timestep
        figsize: Size of entire plot grid
        **kwargs: Additional arguments for plot_provision_evolution_by_node
    """
    # Filter results to only include selected nodes
    filtered_results = []
    for df in results:
        filtered_df = df[df['name'].isin(selected_nodes)].copy()
        filtered_results.append(filtered_df)
    
    plot_provision_evolution_by_node(
        filtered_results, start_year, figsize, **kwargs
    )


def plot_provision_evolution_comparative(
    results_dict: Dict[str, List[pd.DataFrame]],
    start_year: int = 2025,
    figsize_per_scenario: Tuple[int, int] = (25, 15),
    **kwargs
) -> None:
    """
    Plot provision evolution for multiple scenarios side by side
    
    Args:
        results_dict: Dictionary with scenario names as keys and results as values
        start_year: Year of first timestep
        figsize_per_scenario: Size per scenario plot
        **kwargs: Additional arguments for plot_provision_evolution_by_node
    """
    n_scenarios = len(results_dict)
    
    for i, (scenario_name, results) in enumerate(results_dict.items()):
        print(f"\n=== {scenario_name} ===")
        
        # Adjust title to include scenario name
        title = kwargs.get('title', 'Provision Evolution by Node and Month')
        scenario_title = f"{title} - {scenario_name}"
        
        plot_provision_evolution_by_node(
            results, start_year, figsize_per_scenario, 
            title=scenario_title, **{k: v for k, v in kwargs.items() if k != 'title'}
        )


def create_provision_summary_stats(results: List[pd.DataFrame],
                                  name_field: str = "name",
                                  provision_field: str = "provision") -> pd.DataFrame:
    """
    Create summary statistics for provision evolution
    
    Args:
        results: List of DataFrames with provision data
        name_field: Name of the node name column
        provision_field: Name of the provision column
        
    Returns:
        DataFrame with summary statistics by node and month
    """
    node_names = extract_unique_node_names(results, name_field)
    monthly_data = organize_data_by_month(results, node_names, name_field, provision_field)
    
    summary_records = []
    
    for month in range(12):
        for node in node_names:
            values = monthly_data[month][node]
            if values:
                summary_records.append({
                    'month': month + 1,
                    'node': node,
                    'mean_provision': np.mean(values),
                    'std_provision': np.std(values),
                    'min_provision': np.min(values),
                    'max_provision': np.max(values),
                    'n_years': len(values)
                })
    
    return pd.DataFrame(summary_records)


# Example usage functions
def example_basic_evolution_plot():
    """Example of basic provision evolution plot"""
    # plot_provision_evolution_by_node(net.stats.results, START_YEAR)
    pass


def example_custom_styling():
    """Example with custom styling"""
    # plot_provision_evolution_by_node(
    #     net.stats.results,
    #     start_year=START_YEAR,
    #     figsize=(35, 20),
    #     colormap="viridis",
    #     title="Healthcare Provision Evolution Analysis",
    #     colorbar_label="Access Level",
    #     month_names=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
    #                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # )
    pass


def example_subset_analysis():
    """Example of subset analysis"""
    # # Analyze only major cities
    # major_cities = ["Moscow", "St Petersburg", "Novosibirsk", "Ekaterinburg"]
    # plot_provision_evolution_subset(
    #     net.stats.results,
    #     selected_nodes=major_cities,
    #     start_year=START_YEAR,
    #     title="Provision Evolution - Major Cities"
    # )
    pass


def example_comparative_analysis():
    """Example of comparative scenario analysis"""
    # scenarios = {
    #     "Baseline": baseline_results,
    #     "Enhanced Infrastructure": enhanced_results,
    #     "Climate Impact": climate_results
    # }
    # 
    # plot_provision_evolution_comparative(
    #     scenarios,
    #     start_year=START_YEAR,
    #     colormap="RdYlBu_r"
    # )
    pass


def example_summary_statistics():
    """Example of creating summary statistics"""
    # stats = create_provision_summary_stats(net.stats.results)
    # print("Average provision by month:")
    # monthly_avg = stats.groupby('month')['mean_provision'].mean()
    # print(monthly_avg)
    pass


# Legacy function for backward compatibility
def plot_provision_evolution_by_node_legacy(results, start_year=2025, figsize=(30, 18)):
    """Legacy function wrapper - use plot_provision_evolution_by_node instead"""
    plot_provision_evolution_by_node(results, start_year, figsize)