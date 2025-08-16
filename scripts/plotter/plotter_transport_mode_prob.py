import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import pandas as pd


def generate_temperature_range(temp_min: float = -70, temp_max: float = 60, 
                              num_points: int = 2000) -> np.ndarray:
    """
    Generate temperature range for analysis
    
    Args:
        temp_min: Minimum temperature
        temp_max: Maximum temperature
        num_points: Number of temperature points
        
    Returns:
        Array of temperature values
    """
    return np.linspace(temp_min, temp_max, num_points)


def calculate_probability_curves(temps: np.ndarray, transport_modes: List[str],
                               probability_function: Callable[[str, float], float]) -> Dict[str, List[float]]:
    """
    Calculate probability curves for all transport modes
    
    Args:
        temps: Array of temperature values
        transport_modes: List of transport mode names
        probability_function: Function that calculates transport probability
        
    Returns:
        Dictionary with probability curves for each mode
    """
    probability_curves = {}
    
    for mode in transport_modes:
        probs = [probability_function(mode, t) for t in temps]
        probability_curves[mode] = probs
    
    return probability_curves


def find_threshold_intersections(temps: np.ndarray, probs: List[float], 
                               threshold: float) -> List[float]:
    """
    Find temperature points where probability curve intersects threshold
    
    Args:
        temps: Array of temperature values
        probs: Array of probability values
        threshold: Threshold value to find intersections with
        
    Returns:
        List of intersection temperatures
    """
    intersections = []
    
    for i in range(len(temps) - 1):
        if (probs[i] - threshold) * (probs[i + 1] - threshold) <= 0:
            # Linear interpolation to find exact intersection
            t_intersect = temps[i] + (temps[i + 1] - temps[i]) * (threshold - probs[i]) / (probs[i + 1] - probs[i])
            intersections.append(round(t_intersect, 1))
    
    return intersections


def find_operational_ranges(temps: np.ndarray, probs: List[float], 
                          threshold: float) -> List[Tuple[float, float]]:
    """
    Find temperature ranges where probability is above threshold
    
    Args:
        temps: Array of temperature values
        probs: Array of probability values
        threshold: Threshold value
        
    Returns:
        List of (temp_start, temp_end) tuples for operational ranges
    """
    ranges = []
    in_range = False
    start_temp = None
    
    for i, (temp, prob) in enumerate(zip(temps, probs)):
        if prob >= threshold and not in_range:
            # Start of operational range
            start_temp = temp
            in_range = True
        elif prob < threshold and in_range:
            # End of operational range
            if start_temp is not None:
                ranges.append((round(start_temp, 1), round(temps[i-1], 1)))
            in_range = False
            start_temp = None
    
    # Handle case where range extends to end of temperature array
    if in_range and start_temp is not None:
        ranges.append((round(start_temp, 1), round(temps[-1], 1)))
    
    return ranges


def find_all_operational_ranges(temps: np.ndarray, probability_curves: Dict[str, List[float]],
                               threshold: float) -> Dict[str, List[Tuple[float, float]]]:
    """
    Find operational temperature ranges for all transport modes
    
    Args:
        temps: Array of temperature values
        probability_curves: Dictionary with probability curves
        threshold: Threshold value
        
    Returns:
        Dictionary with operational ranges for each mode
    """
    operational_ranges = {}
    
    for mode, probs in probability_curves.items():
        ranges = find_operational_ranges(temps, probs, threshold)
        if ranges:
            operational_ranges[mode] = ranges
    
    return operational_ranges


def find_all_threshold_intersections(temps: np.ndarray, probability_curves: Dict[str, List[float]],
                                   threshold: float) -> Dict[str, List[float]]:
    """
    Find threshold intersections for all transport modes
    
    Args:
        temps: Array of temperature values
        probability_curves: Dictionary with probability curves
        threshold: Threshold value
        
    Returns:
        Dictionary with ALL intersection temperatures for each mode (not just first)
    """
    threshold_temperatures = {}
    
    for mode, probs in probability_curves.items():
        intersections = find_threshold_intersections(temps, probs, threshold)
        if intersections:
            threshold_temperatures[mode] = intersections  # Keep ALL intersections
    
    return threshold_temperatures


def plot_probability_curve(ax, temps: np.ndarray, probs: List[float], 
                         mode: str, color: str, linewidth: float = 2) -> Any:
    """
    Plot probability curve for a single transport mode
    
    Args:
        ax: Matplotlib axis
        temps: Temperature array
        probs: Probability array
        mode: Transport mode name
        color: Line color
        linewidth: Line width
        
    Returns:
        Line object from plot
    """
    line, = ax.plot(temps, probs, label=mode, linewidth=linewidth, color=color)
    return line


def annotate_intersection(ax, temp: float, threshold: float, mode: str, 
                        line_color: str, font_size: int = 12,
                        annotation_offset: Tuple[int, int] = (10, 10)) -> None:
    """
    Add annotation for threshold intersection point
    
    Args:
        ax: Matplotlib axis
        temp: Intersection temperature
        threshold: Threshold value
        mode: Transport mode name
        line_color: Color of the line for arrow
        font_size: Font size for annotation
        annotation_offset: Offset for annotation text
    """
    # Plot intersection point
    ax.plot(temp, threshold, "ko", markersize=8)
    
    # Add annotation
    ax.annotate(
        f"{temp:.1f}°C",
        (temp, threshold),
        xytext=annotation_offset,
        textcoords="offset points",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color=line_color),
        fontsize=font_size,
    )


def add_threshold_shading(ax, temps: np.ndarray, threshold: float,
                        shade_color: str = "grey", shade_alpha: float = 0.2,
                        shade_label: str = "Non-operational zone") -> None:
    """
    Add shading below threshold line
    
    Args:
        ax: Matplotlib axis
        temps: Temperature array
        threshold: Threshold value
        shade_color: Color for shading
        shade_alpha: Transparency of shading
        shade_label: Label for shaded area
    """
    ax.fill_between(temps, 0, threshold, color=shade_color, alpha=shade_alpha, label=shade_label)


def add_threshold_line(ax, threshold: float, line_color: str = "r",
                      line_style: str = "--", line_alpha: float = 0.5) -> None:
    """
    Add horizontal threshold line
    
    Args:
        ax: Matplotlib axis
        threshold: Threshold value
        line_color: Line color
        line_style: Line style
        line_alpha: Line transparency
    """
    ax.axhline(
        y=threshold, color=line_color, linestyle=line_style, 
        alpha=line_alpha, label=f"Threshold ({threshold})"
    )


def configure_plot_styling(ax, title: str, xlabel: str = "Temperature (°C)",
                         ylabel: str = "Transport Probability", 
                         font_size: int = 12, temp_range: Tuple[float, float] = (-70, 60),
                         legend_position: Tuple[float, float] = (1.05, 1),
                         grid_alpha: float = 0.3) -> None:
    """
    Configure plot styling and labels
    
    Args:
        ax: Matplotlib axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        font_size: Font size
        temp_range: Temperature range for x-axis
        legend_position: Legend position
        grid_alpha: Grid transparency
    """
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title, fontsize=font_size, pad=20)
    ax.legend(bbox_to_anchor=legend_position, loc="upper left", fontsize=font_size)
    ax.grid(True, alpha=grid_alpha)
    ax.set_xlim(temp_range)


def plot_transport_probability_analysis(
    transport_modes: List[str],
    transport_modes_color: Dict[str, str],
    probability_function: Callable[[str, float], float],
    threshold: float,
    temp_min: float = -70,
    temp_max: float = 60,
    num_temp_points: int = 2000,
    figsize: Tuple[int, int] = (14, 4),
    font_size: int = 12,
    linewidth: float = 2,
    title: str = "Transport Probability vs Temperature\nwith Threshold Intersections",
    xlabel: str = "Temperature (°C)",
    ylabel: str = "Transport Probability",
    show_intersections: bool = True,
    show_threshold_shading: bool = True,
    show_threshold_line: bool = True,
    annotation_offset: Tuple[int, int] = (10, 10),
    shade_color: str = "grey",
    shade_alpha: float = 0.2,
    grid_alpha: float = 0.3,
    legend_position: Tuple[float, float] = (1.05, 1)
) -> Tuple[Dict[str, float], plt.Figure, plt.Axes]:
    """
    Create complete transport probability analysis plot
    
    Args:
        transport_modes: List of transport mode names
        transport_modes_color: Dictionary mapping modes to colors
        probability_function: Function to calculate transport probabilities
        threshold: Probability threshold value
        temp_min: Minimum temperature
        temp_max: Maximum temperature
        num_temp_points: Number of temperature points
        figsize: Figure size
        font_size: Font size for labels
        linewidth: Line width for curves
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_intersections: Whether to show intersection annotations
        show_threshold_shading: Whether to show shading below threshold
        show_threshold_line: Whether to show threshold line
        annotation_offset: Offset for intersection annotations
        shade_color: Color for threshold shading
        shade_alpha: Transparency of shading
        grid_alpha: Grid transparency
        legend_position: Position of legend
        
    Returns:
        Tuple of (threshold_temperatures_dict, figure, axes)
    """
    # Generate temperature range
    temps = generate_temperature_range(temp_min, temp_max, num_temp_points)
    
    # Calculate probability curves
    probability_curves = calculate_probability_curves(temps, transport_modes, probability_function)
    
    # Find threshold intersections
    threshold_temperatures = find_all_threshold_intersections(temps, probability_curves, threshold)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot probability curves and intersections
    for mode in transport_modes:
        probs = probability_curves[mode]
        color = transport_modes_color.get(mode, 'blue')
        
        # Plot curve
        line = plot_probability_curve(ax, temps, probs, mode, color, linewidth)
        
        # Add intersection annotation if requested and intersection exists
        if show_intersections and mode in threshold_temperatures:
            temp_intersect = threshold_temperatures[mode]
            annotate_intersection(
                ax, temp_intersect, threshold, mode, color, 
                font_size, annotation_offset
            )
    
    # Add threshold elements
    if show_threshold_shading:
        add_threshold_shading(ax, temps, threshold, shade_color, shade_alpha)
    
    if show_threshold_line:
        add_threshold_line(ax, threshold)
    
    # Configure styling
    configure_plot_styling(
        ax, title, xlabel, ylabel, font_size, 
        (temp_min, temp_max), legend_position, grid_alpha
    )
    
    plt.tight_layout()
    plt.show()
    
    return threshold_temperatures, fig, ax


def create_threshold_sensitivity_analysis(
    transport_modes: List[str],
    transport_modes_color: Dict[str, str],
    probability_function: Callable[[str, float], float],
    threshold_values: List[float],
    temp_min: float = -70,
    temp_max: float = 60,
    figsize: Tuple[int, int] = (16, 10)
) -> Dict[float, Dict[str, float]]:
    """
    Analyze sensitivity to different threshold values
    
    Args:
        transport_modes: List of transport mode names
        transport_modes_color: Dictionary mapping modes to colors
        probability_function: Function to calculate transport probabilities
        threshold_values: List of threshold values to analyze
        temp_min: Minimum temperature
        temp_max: Maximum temperature
        figsize: Figure size
        
    Returns:
        Dictionary with threshold intersections for each threshold value
    """
    temps = generate_temperature_range(temp_min, temp_max, 2000)
    probability_curves = calculate_probability_curves(temps, transport_modes, probability_function)
    
    # Create subplot grid
    n_thresholds = len(threshold_values)
    cols = min(3, n_thresholds)
    rows = (n_thresholds + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_thresholds == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    all_threshold_temperatures = {}
    
    for i, threshold in enumerate(threshold_values):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Find intersections for this threshold
        threshold_temperatures = find_all_threshold_intersections(
            temps, probability_curves, threshold
        )
        all_threshold_temperatures[threshold] = threshold_temperatures
        
        # Plot curves
        for mode in transport_modes:
            probs = probability_curves[mode]
            color = transport_modes_color.get(mode, 'blue')
            ax.plot(temps, probs, label=mode, linewidth=1.5, color=color)
            
            # Add intersection point
            if mode in threshold_temperatures:
                temp_intersect = threshold_temperatures[mode]
                ax.plot(temp_intersect, threshold, "ko", markersize=6)
        
        # Add threshold line and styling
        ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.7)
        ax.fill_between(temps, 0, threshold, color="grey", alpha=0.1)
        
        ax.set_title(f"Threshold: {threshold}")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)
        ax.set_xlim((temp_min, temp_max))
        
        if i == 0:  # Add legend to first subplot
            ax.legend(loc="upper right", fontsize=10)
    
    # Hide unused subplots
    for j in range(i + 1, rows * cols):
        row, col = j // cols, j % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle("Threshold Sensitivity Analysis", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return all_threshold_temperatures


def create_intersection_summary_table(threshold_temperatures: Dict[str, float]) -> pd.DataFrame:
    """
    Create summary table of threshold intersections
    
    Args:
        threshold_temperatures: Dictionary of mode -> intersection temperature
        
    Returns:
        DataFrame with intersection summary
    """
    if not threshold_temperatures:
        return pd.DataFrame(columns=['Transport Mode', 'Intersection Temperature (°C)'])
    
    summary_data = [
        {'Transport Mode': mode, 'Intersection Temperature (°C)': temp}
        for mode, temp in threshold_temperatures.items()
    ]
    
    return pd.DataFrame(summary_data).sort_values('Intersection Temperature (°C)')


# Example usage functions
def example_basic_analysis():
    """Example of basic transport probability analysis"""
    # def get_transport_probability(mode, temp):
    #     # Your probability calculation logic
    #     return 0.5  # Placeholder
    # 
    # transport_modes = ["car_warm", "car_cold", "plane", "water_ship", "winter_tr"]
    # transport_modes_color = {
    #     "car_warm": "red",
    #     "car_cold": "blue", 
    #     "plane": "green",
    #     "water_ship": "orange",
    #     "winter_tr": "purple"
    # }
    # 
    # threshold_temps, fig, ax = plot_transport_probability_analysis(
    #     transport_modes=transport_modes,
    #     transport_modes_color=transport_modes_color,
    #     probability_function=get_transport_probability,
    #     threshold=0.5
    # )
    # 
    # print("Threshold intersection temperatures:")
    # for mode, temp in threshold_temps.items():
    #     print(f"{mode}: {temp}°C")
    pass


def example_sensitivity_analysis():
    """Example of threshold sensitivity analysis"""
    # threshold_values = [0.3, 0.5, 0.7, 0.9]
    # sensitivity_results = create_threshold_sensitivity_analysis(
    #     transport_modes=transport_modes,
    #     transport_modes_color=transport_modes_color,
    #     probability_function=get_transport_probability,
    #     threshold_values=threshold_values
    # )
    # 
    # # Create summary
    # for threshold, intersections in sensitivity_results.items():
    #     print(f"\nThreshold {threshold}:")
    #     summary_table = create_intersection_summary_table(intersections)
    #     print(summary_table)
    pass


def example_custom_styling():
    """Example with custom styling"""
    # threshold_temps, fig, ax = plot_transport_probability_analysis(
    #     transport_modes=transport_modes,
    #     transport_modes_color=transport_modes_color,
    #     probability_function=get_transport_probability,
    #     threshold=0.5,
    #     figsize=(18, 6),
    #     title="Arctic Transport Viability Analysis",
    #     temp_min=-80,
    #     temp_max=40,
    #     shade_color="lightblue",
    #     shade_alpha=0.3,
    #     font_size=14,
    #     linewidth=3
    # )
    pass


# Legacy function for backward compatibility
def plot_transport_probability_legacy(transport_modes, transport_modes_color, 
                                    get_transport_probability, threshold, 
                                    temps=None, font_size=12):
    """
    Legacy function wrapper that returns operational temperature ranges
    
    Returns:
        Dictionary mapping transport modes to lists of (temp_from, temp_to) tuples
        Example: {
            'Aviation': [(-59.7, -39.2), (15.3, 42.8)],
            'Regular road': [(-69.1, -45.2), (8.7, 38.9)],
            'Water transport': [(2.1, 47.3)]
        }
    """
    if temps is None:
        temps = np.linspace(-70, 60, 2000)
    
    # Calculate probability curves
    probability_curves = calculate_probability_curves(temps, transport_modes, get_transport_probability)
    
    # Find intersection points for annotations
    threshold_temperatures = find_all_threshold_intersections(temps, probability_curves, threshold)
    
    # Find operational ranges for return value
    operational_ranges = find_all_operational_ranges(temps, probability_curves, threshold)
    
    # Create figure with original behavior
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Plot probability curves and intersections
    for mode in transport_modes:
        probs = probability_curves[mode]
        color = transport_modes_color.get(mode, 'blue')
        
        # Plot curve
        line = plot_probability_curve(ax, temps, probs, mode, color, linewidth=2)
        
        # Add intersection annotations for ALL intersections
        if mode in threshold_temperatures:
            intersections = threshold_temperatures[mode]
            if isinstance(intersections, list):
                # Multiple intersections
                for temp_intersect in intersections:
                    annotate_intersection(
                        ax, temp_intersect, threshold, mode, color, 
                        font_size, (10, 10)
                    )
            else:
                # Single intersection (backward compatibility)
                annotate_intersection(
                    ax, intersections, threshold, mode, color, 
                    font_size, (10, 10)
                )
    
    # Add threshold elements (original behavior)
    add_threshold_shading(ax, temps, threshold, "grey", 0.2, "Non-operational zone")
    add_threshold_line(ax, threshold, "r", "--", 0.5)
    
    # Configure styling with original parameters
    ax.set_xlabel("Temperature (°C)", fontsize=font_size)
    ax.set_ylabel("Transport Probability", fontsize=font_size)
    ax.set_title(
        "Transport Probability vs Temperature\nwith Threshold Intersections",
        fontsize=font_size, pad=20
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=font_size)
    ax.grid(True, alpha=0.3)
    ax.set_xlim((-70, 60))  # Force the full temperature range
    ax.set_ylim((-0.05, 1.05))  # Ensure full probability range is visible
    
    plt.tight_layout()
    plt.show()
    
    # Return operational ranges instead of just intersection points
    return operational_ranges