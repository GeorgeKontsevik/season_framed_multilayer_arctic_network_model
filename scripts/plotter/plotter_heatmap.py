import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scicolor  # Assuming this is the scicolor import
from typing import List, Optional, Tuple


def prepare_categorical_months(df: pd.DataFrame, month_column: str, 
                              month_order: List[str]) -> pd.DataFrame:
    """
    Convert month column to categorical with specified order
    
    Args:
        df: DataFrame containing the data
        month_column: Name of the month column
        month_order: List of months in desired order
        
    Returns:
        DataFrame with categorical month column
    """
    df_copy = df.copy()
    df_copy[month_column] = pd.Categorical(
        df_copy[month_column], categories=month_order, ordered=True
    )
    return df_copy


def create_transport_mode_pivot(df: pd.DataFrame, mode: str, 
                               month_column: str = "month_name",
                               year_column: str = "year",
                               count_column: str = "count") -> pd.DataFrame:
    """
    Create pivot table for a specific transport mode
    
    Args:
        df: DataFrame containing the data
        mode: Transport mode to filter by
        month_column: Name of the month column
        year_column: Name of the year column
        count_column: Name of the count column
        
    Returns:
        Pivot table with months as index and years as columns
    """
    df_mode = df[df["mode"] == mode]
    pivot = df_mode.pivot(index=month_column, columns=year_column, values=count_column)
    return pivot


def plot_transport_heatmap(pivot: pd.DataFrame, mode: str, 
                          max_nodes: int,
                          figsize: Tuple[int, int] = (14, 3),
                          colormap: str = "Hokusai2",
                          linewidths: float = 0.5,
                          title_template: str = 'Number of nodes when "{mode}" operates',
                          xlabel: str = "Год",
                          ylabel: str = "Месяц") -> None:
    """
    Create and display heatmap for transport mode data
    
    Args:
        pivot: Pivot table with data to plot
        mode: Transport mode name for title
        max_nodes: Maximum number of nodes for color scale
        figsize: Figure size tuple
        colormap: Colormap name
        linewidths: Width of lines between cells
        title_template: Template for plot title with {mode} placeholder
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    plt.figure(figsize=figsize)
    
    # Get colormap - handle different science plot versions
    try:
        cmap = scicolor.get_cmap(colormap)
    except AttributeError:
        # Fallback for different scienceplots versions
        cmap = colormap
    
    sns.heatmap(
        pivot,
        cmap=cmap,
        linewidths=linewidths,
        vmin=0,
        vmax=max_nodes,
    )
    
    plt.title(title_template.format(mode=mode))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def create_transport_mode_heatmaps(
    df_modes_monthly: pd.DataFrame,
    transport_modes: List[str],
    month_order: List[str],
    max_nodes: int,
    month_column: str = "month_name",
    year_column: str = "year",
    count_column: str = "count",
    mode_column: str = "mode",
    figsize: Tuple[int, int] = (14, 3),
    colormap: str = "Hokusai2",
    linewidths: float = 0.5,
    title_template: str = 'Number of nodes when "{mode}" operates',
    xlabel: str = "Год",
    ylabel: str = "Месяц",
    invert_data: bool = False  # NEW: Simple flag to invert the data
) -> None:
    """
    Create heatmaps for all transport modes
    
    Args:
        df_modes_monthly: DataFrame containing monthly transport mode data
        transport_modes: List of transport modes to plot
        month_order: List of months in desired order
        max_nodes: Maximum number of nodes (from G_undirected.nodes)
        month_column: Name of the month column
        year_column: Name of the year column
        count_column: Name of the count column
        mode_column: Name of the mode column
        figsize: Figure size tuple
        colormap: Colormap name (e.g., "Hokusai2")
        linewidths: Width of lines between cells
        title_template: Template for plot title with {mode} placeholder
        xlabel: X-axis label
        ylabel: Y-axis label
        invert_data: If True, show max_nodes - count (flip operational/non-operational)
    """
    # Prepare categorical months
    df_prepared = prepare_categorical_months(df_modes_monthly, month_column, month_order)
    
    # Create heatmaps for each transport mode
    for mode in transport_modes:
        # Create pivot table for current mode
        pivot = create_transport_mode_pivot(
            df_prepared, mode, month_column, year_column, count_column
        )
        
        # Invert data if requested
        if invert_data:
            display_data = max_nodes - pivot
            # Update title to reflect the inversion
            if 'operates' in title_template:
                current_title = title_template.replace('operates', 'does not operate').format(mode=mode)
            elif 'operational' in title_template.lower():
                current_title = title_template.replace('operational', 'non-operational').format(mode=mode)
            else:
                current_title = f"Inverted: {title_template}".format(mode=mode)
        else:
            display_data = pivot
            current_title = title_template.format(mode=mode)
        
        # Plot heatmap
        plot_transport_heatmap(
            pivot=display_data,
            mode=mode,
            max_nodes=max_nodes,
            figsize=figsize,
            colormap=colormap,
            linewidths=linewidths,
            title_template=current_title,
            xlabel=xlabel,
            ylabel=ylabel
        )


# Example usage:
def example_usage():
    """
    Example of how to use the transport mode heatmap functions
    """
    # Example month order (adjust based on your data)
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    # Example transport modes
    transport_modes = ["car_warm", "car_cold", "plane", "water_ship", "winter_tr"]
    
    # Assuming you have your data loaded:
    # df_modes_monthly = your_dataframe
    # G_undirected = your_graph
    
    # Create all heatmaps
    # create_transport_mode_heatmaps(
    #     df_modes_monthly=df_modes_monthly,
    #     transport_modes=transport_modes,
    #     month_order=month_order,
    #     max_nodes=len(G_undirected.nodes)
    # )
    
    # Or create individual heatmaps with custom parameters
    # for mode in transport_modes:
    #     df_prepared = prepare_categorical_months(df_modes_monthly, "month_name", month_order)
    #     pivot = create_transport_mode_pivot(df_prepared, mode)
    #     plot_transport_heatmap(pivot, mode, len(G_undirected.nodes))


# Legacy function for backward compatibility
def create_heatmaps_legacy(df_modes_monthly, transport_modes, month_order, G_undirected):
    """Legacy function wrapper - use create_transport_mode_heatmaps instead"""
    create_transport_mode_heatmaps(
        df_modes_monthly=df_modes_monthly,
        transport_modes=transport_modes,
        month_order=month_order,
        max_nodes=len(G_undirected.nodes)
    )