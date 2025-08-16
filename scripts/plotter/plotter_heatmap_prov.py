import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scicolor  # Assuming this is the scicolor import
from typing import List, Optional, Tuple, Union
import numpy as np


def extract_provision_records(results: List[pd.DataFrame], 
                            start_year: int = 1982,
                            provision_field: str = "provision",
                            name_field: str = "name") -> List[dict]:
    """
    Extract provision records from time series results
    
    Args:
        results: List of DataFrames with provision data
        start_year: Starting year for time series
        provision_field: Name of the provision column
        name_field: Name of the location/node name column
        
    Returns:
        List of dictionaries with year, month, name, and provision data
    """
    records = []
    
    for i, df in enumerate(results):
        year = start_year + (i // 12)
        month = (i % 12) + 1  # 1-12
        
        for _, row in df.reset_index().iterrows():
            records.append({
                "year": year,
                "month": month,
                "name": row[name_field],
                "provision": row[provision_field],
            })
    
    return records


def create_provision_dataframe(records: List[dict], 
                             zero_threshold: float = 0.0) -> pd.DataFrame:
    """
    Create DataFrame with provision flags and month names
    
    Args:
        records: List of provision records
        zero_threshold: Threshold below which provision is considered zero
        
    Returns:
        DataFrame with provision analysis columns
    """
    df_all = pd.DataFrame(records)
    
    # Create binary flags for zero/non-zero provision
    df_all["is_non_zero"] = df_all["provision"] > zero_threshold
    df_all["is_zero"] = df_all["provision"] <= zero_threshold
    
    # Add month names
    df_all["month_name"] = pd.to_datetime(df_all["month"], format="%m").dt.strftime("%b")
    
    return df_all


def calculate_provision_counts(df_all: pd.DataFrame, 
                             group_by_cols: List[str] = None) -> pd.DataFrame:
    """
    Calculate provision counts and percentages by grouping variables
    
    Args:
        df_all: DataFrame with provision data
        group_by_cols: Columns to group by (default: ["year", "month_name"])
        
    Returns:
        DataFrame with provision counts and percentages
    """
    if group_by_cols is None:
        group_by_cols = ["year", "month_name"]
    
    # Calculate counts
    counts = (
        df_all.groupby(group_by_cols)["is_non_zero"]
        .agg(non_zero="sum", total="count")
        .reset_index()
    )
    
    # Calculate derived metrics
    counts["zero"] = counts["total"] - counts["non_zero"]
    counts["zero_pct"] = 100 * counts["zero"] / counts["total"]
    counts["non_zero_pct"] = 100 * counts["non_zero"] / counts["total"]
    
    return counts


def add_categorical_months(df: pd.DataFrame, 
                         month_column: str = "month_name",
                         month_order: List[str] = None) -> pd.DataFrame:
    """
    Add categorical month column with proper ordering
    
    Args:
        df: DataFrame with month names
        month_column: Name of the month column
        month_order: Custom month order (default: Jan-Dec abbreviations)
        
    Returns:
        DataFrame with categorical month column
    """
    if month_order is None:
        month_order = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]
    
    df = df.copy()
    df[month_column] = pd.Categorical(
        df[month_column], categories=month_order, ordered=True
    )
    
    return df


def create_provision_heatmap(pivot_data: pd.DataFrame,
                           title: str = "Provision Analysis",
                           figsize: Tuple[int, int] = (14, 2),
                           colormap: str = "Hokusai2",
                           linewidths: float = 0.5,
                           annotate: bool = False,
                           annotation_format: str = ".1f",
                           vmin: Optional[float] = None,
                           vmax: Optional[float] = None,
                           xlabel: str = "Year",
                           ylabel: str = "Month",
                           cbar_label: Optional[str] = None) -> None:
    """
    Create heatmap for provision data
    
    Args:
        pivot_data: Pivot table with data to plot
        title: Plot title
        figsize: Figure size tuple
        colormap: Colormap name
        linewidths: Width of lines between cells
        annotate: Whether to show values in cells
        annotation_format: Format string for annotations
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        xlabel: X-axis label
        ylabel: Y-axis label
        cbar_label: Colorbar label
    """
    plt.figure(figsize=figsize)
    
    # Get colormap - handle different science plot versions
    try:
        cmap = scicolor.get_cmap(colormap)
    except AttributeError:
        cmap = colormap
    
    # Create heatmap
    heatmap = sns.heatmap(
        pivot_data,
        cmap=cmap,
        linewidths=linewidths,
        annot=annotate,
        fmt=annotation_format,
        vmin=vmin,
        vmax=vmax,
    )
    
    # Add colorbar label if specified
    if cbar_label:
        heatmap.collections[0].colorbar.set_label(cbar_label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def count_provision_zero_nonzero(results: List[pd.DataFrame], 
                                start_year: int = 1982,
                                provision_field: str = "provision",
                                name_field: str = "name",
                                zero_threshold: float = 0.0,
                                month_order: List[str] = None) -> pd.DataFrame:
    """
    Count zero and non-zero provision by month and year
    
    Args:
        results: List of DataFrames with provision data
        start_year: Starting year for time series
        provision_field: Name of the provision column
        name_field: Name of the location/node name column
        zero_threshold: Threshold below which provision is considered zero
        month_order: Custom month order (default: Jan-Dec abbreviations)
        
    Returns:
        DataFrame with provision counts and percentages
    """
    # Extract records
    records = extract_provision_records(results, start_year, provision_field, name_field)
    
    # Create DataFrame with flags
    df_all = create_provision_dataframe(records, zero_threshold)
    
    # Calculate counts
    counts = calculate_provision_counts(df_all)
    
    # Add categorical months
    counts = add_categorical_months(counts, month_order=month_order)
    
    return counts


def analyze_provision_patterns(results: List[pd.DataFrame],
                             start_year: int = 1982,
                             provision_field: str = "provision",
                             name_field: str = "name",
                             zero_threshold: float = 0.0,
                             figsize: Tuple[int, int] = (14, 2),
                             colormap: str = "Hokusai2",
                             show_zero_pct: bool = True,
                             show_zero_count: bool = False,
                             show_non_zero_pct: bool = False,
                             vmax_pct: float = 20.0,
                             title_template: str = "Share of nodes with zero provision (%), by month and year") -> pd.DataFrame:
    """
    Complete analysis and visualization of provision patterns
    
    Args:
        results: List of DataFrames with provision data
        start_year: Starting year for time series
        provision_field: Name of the provision column
        name_field: Name of the location/node name column
        zero_threshold: Threshold below which provision is considered zero
        figsize: Figure size for plots
        colormap: Colormap for heatmaps
        show_zero_pct: Whether to show zero provision percentage heatmap
        show_zero_count: Whether to show zero provision count heatmap
        show_non_zero_pct: Whether to show non-zero provision percentage heatmap
        vmax_pct: Maximum value for percentage scale
        title_template: Template for plot titles
        
    Returns:
        DataFrame with provision analysis results
    """
    # Get provision counts
    counts = count_provision_zero_nonzero(
        results, start_year, provision_field, name_field, zero_threshold
    )
    
    # Create heatmaps based on flags
    if show_zero_pct:
        pivot_zero_pct = counts.pivot(index="month_name", columns="year", values="zero_pct")
        create_provision_heatmap(
            pivot_zero_pct,
            title=title_template,
            figsize=figsize,
            colormap=colormap,
            vmax=vmax_pct,
            cbar_label="Zero Provision (%)"
        )
    
    if show_zero_count:
        pivot_zero = counts.pivot(index="month_name", columns="year", values="zero")
        create_provision_heatmap(
            pivot_zero,
            title="Count of nodes with zero provision, by month and year",
            figsize=figsize,
            colormap=colormap,
            cbar_label="Zero Provision Count"
        )
    
    if show_non_zero_pct:
        pivot_non_zero_pct = counts.pivot(index="month_name", columns="year", values="non_zero_pct")
        create_provision_heatmap(
            pivot_non_zero_pct,
            title="Share of nodes with non-zero provision (%), by month and year",
            figsize=figsize,
            colormap=colormap,
            vmin=80.0,  # Focus on high-provision areas
            vmax=100.0,
            cbar_label="Non-Zero Provision (%)"
        )
    
    return counts


def compare_provision_thresholds(results: List[pd.DataFrame],
                               thresholds: List[float] = [0.0, 0.1, 0.5, 1.0],
                               start_year: int = 1982,
                               provision_field: str = "provision",
                               figsize: Tuple[int, int] = (16, 8)) -> None:
    """
    Compare provision patterns across different thresholds
    
    Args:
        results: List of DataFrames with provision data
        thresholds: List of thresholds to compare
        start_year: Starting year for time series
        provision_field: Name of the provision column
        figsize: Figure size for subplot grid
    """
    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(2, (n_thresholds + 1) // 2, figsize=figsize)
    axes = axes.flatten() if n_thresholds > 1 else [axes]
    
    for i, threshold in enumerate(thresholds):
        counts = count_provision_zero_nonzero(
            results, start_year, provision_field, zero_threshold=threshold
        )
        
        pivot_data = counts.pivot(index="month_name", columns="year", values="zero_pct")
        
        sns.heatmap(
            pivot_data,
            ax=axes[i],
            cmap="Reds",
            linewidths=0.5,
            cbar=True,
            vmax=50
        )
        
        axes[i].set_title(f"Zero provision threshold: {threshold}")
        axes[i].set_xlabel("Year")
        axes[i].set_ylabel("Month")
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage functions
def example_basic_analysis():
    """Example of basic provision analysis"""
    # counts = count_provision_zero_nonzero(net.stats.results, start_year=START_YEAR)
    # 
    # # Create zero percentage heatmap
    # pivot_zero_pct = counts.pivot(index="month_name", columns="year", values="zero_pct")
    # create_provision_heatmap(
    #     pivot_zero_pct,
    #     title="Share of nodes with zero provision (%), by month and year",
    #     vmax=20
    # )
    pass


def example_comprehensive_analysis():
    """Example of comprehensive provision analysis"""
    # analyze_provision_patterns(
    #     results=net.stats.results,
    #     start_year=START_YEAR,
    #     show_zero_pct=True,
    #     show_zero_count=True,
    #     show_non_zero_pct=True,
    #     vmax_pct=25.0
    # )
    pass


def example_threshold_comparison():
    """Example of threshold comparison"""
    # compare_provision_thresholds(
    #     results=net.stats.results,
    #     thresholds=[0.0, 0.25, 0.5, 0.75],
    #     start_year=START_YEAR
    # )
    pass


def example_custom_analysis():
    """Example of custom provision analysis"""
    # # Custom analysis for specific service
    # counts = count_provision_zero_nonzero(
    #     results=hospital_results,
    #     start_year=1990,
    #     provision_field="hospital_provision",
    #     name_field="settlement_name",
    #     zero_threshold=0.1  # Consider anything below 10% as effectively zero
    # )
    # 
    # # Create custom visualization
    # pivot_data = counts.pivot(index="month_name", columns="year", values="zero_pct")
    # create_provision_heatmap(
    #     pivot_data,
    #     title="Healthcare Access Gaps (< 10% provision)",
    #     colormap="OrRd",
    #     figsize=(20, 6),
    #     vmax=30,
    #     annotate=True,
    #     annotation_format=".0f"
    # )
    pass


# Legacy function for backward compatibility
def count_provision_zero_nonzero_legacy(results, start_year=1982):
    """Legacy function wrapper - use count_provision_zero_nonzero instead"""
    return count_provision_zero_nonzero(results, start_year)