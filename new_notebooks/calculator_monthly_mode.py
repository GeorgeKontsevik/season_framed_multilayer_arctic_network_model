import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import networkx as nx


def is_temperature_in_operational_ranges(temp: float, ranges: List[Tuple[float, float]]) -> bool:
    """
    Check if temperature is within any of the operational ranges
    
    Args:
        temp: Temperature value to check (scalar)
        ranges: List of (temp_from, temp_to) tuples
        
    Returns:
        True if temperature is in any operational range
    """
    # Ensure temp is a scalar value
    if hasattr(temp, 'item'):
        temp = temp.item()  # Convert numpy scalar to Python scalar
    elif hasattr(temp, '__len__') and len(temp) == 1:
        temp = temp[0]  # Extract from single-element array
    elif hasattr(temp, '__iter__') and not isinstance(temp, str):
        # If it's an array, take the first element
        temp = next(iter(temp))
    
    # Debug: print types if there's an issue
    if not isinstance(temp, (int, float)):
        print(f"Warning: temp is not a number: {temp} (type: {type(temp)})")
        return False
    
    for range_tuple in ranges:
        # Ensure we have a proper tuple
        if not isinstance(range_tuple, (tuple, list)) or len(range_tuple) != 2:
            print(f"Warning: Invalid range format: {range_tuple} (type: {type(range_tuple)})")
            continue
            
        temp_from, temp_to = range_tuple
        
        # Ensure range bounds are numbers
        if not isinstance(temp_from, (int, float)) or not isinstance(temp_to, (int, float)):
            print(f"Warning: Range bounds not numbers: {temp_from} ({type(temp_from)}), {temp_to} ({type(temp_to)})")
            continue
            
        if temp_from <= temp <= temp_to:
            return True
    return False


def extract_temperature_at_time(temp_data, time_index: int) -> float:
    """
    Safely extract temperature value at a specific time index
    
    Args:
        temp_data: Temperature data (could be array, Series, etc.)
        time_index: Time index to extract
        
    Returns:
        Scalar temperature value
    """
    if hasattr(temp_data, 'values'):
        # Pandas Series
        temp = temp_data.values[time_index]
    elif hasattr(temp_data, 'iloc'):
        # Pandas Series with iloc
        temp = temp_data.iloc[time_index]
    else:
        # Numpy array or similar
        temp = temp_data[time_index]
    
    # Ensure it's a scalar
    if hasattr(temp, 'item'):
        return temp.item()
    elif hasattr(temp, '__len__') and len(temp) == 1:
        return temp[0]
    else:
        return float(temp)


def count_operational_nodes_by_mode(G: nx.Graph, time_index: int, 
                                  operational_ranges: Dict[str, List[Tuple[float, float]]],
                                  temperature_field: str = "temperature") -> Dict[str, int]:
    """
    Count operational nodes for each transport mode at a specific time
    
    Args:
        G: NetworkX graph with temperature data
        time_index: Time index to analyze
        operational_ranges: Dictionary of operational ranges for each mode
        temperature_field: Name of temperature field in node data
        
    Returns:
        Dictionary with counts for each transport mode
    """
    mode_counts = {mode: 0 for mode in operational_ranges.keys()}
    
    for node_id, data in G.nodes(data=True):
        if temperature_field in data:
            # Safely extract scalar temperature value
            temp = extract_temperature_at_time(data[temperature_field], time_index)
            
            for mode, ranges in operational_ranges.items():
                if is_temperature_in_operational_ranges(temp, ranges):
                    mode_counts[mode] += 1
    
    return mode_counts


def create_monthly_transport_analysis(
    G: nx.Graph,
    operational_ranges: Dict[str, List[Tuple[float, float]]],
    start_year: int,
    months_in_year: int = 12,
    temperature_field: str = "temperature"
) -> pd.DataFrame:
    """
    Create monthly analysis of operational transport modes
    
    Args:
        G: NetworkX graph with temperature data
        operational_ranges: Dictionary of operational ranges for each mode
        start_year: Starting year of the analysis
        months_in_year: Number of months per year
        temperature_field: Name of temperature field in node data
        
    Returns:
        DataFrame with monthly transport mode counts
    """
    # Extract time dimension
    sample_node = list(G.nodes(data=True))[0]
    T = sample_node[1][temperature_field].shape[0]
    n_years = T // months_in_year
    
    records = []
    
    for t in range(T):
        year = start_year + (t // months_in_year)
        month = (t % months_in_year) + 1
        
        # Count operational nodes for each mode at this time point
        mode_counts = count_operational_nodes_by_mode(
            G, t, operational_ranges, temperature_field
        )
        
        # Create records for each mode
        for mode, count in mode_counts.items():
            records.append({
                "year": year,
                "month": month,
                "month_name": pd.to_datetime(str(month), format="%m").strftime("%b"),
                "mode": mode,
                "count": count,
            })
    
    return pd.DataFrame(records)


def create_transport_analysis_with_probability_function(
    G: nx.Graph,
    transport_modes: List[str],
    transport_modes_color: Dict[str, str],
    probability_function: callable,
    threshold: float,
    start_year: int,
    months_in_year: int = 12,
    temperature_field: str = "temperature",
    temp_range: Tuple[float, float] = (-70, 60),
    num_temp_points: int = 2000
) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[float, float]]]]:
    """
    Complete transport analysis workflow
    
    Args:
        G: NetworkX graph with temperature data
        transport_modes: List of transport mode names
        transport_modes_color: Dictionary mapping modes to colors
        probability_function: Function to calculate transport probabilities
        threshold: Probability threshold
        start_year: Starting year of analysis
        months_in_year: Number of months per year
        temperature_field: Name of temperature field in node data
        temp_range: Temperature range for probability analysis
        num_temp_points: Number of temperature points for analysis
        
    Returns:
        Tuple of (monthly_dataframe, operational_ranges_dict)
    """
    from .transport_probability_visualization import plot_transport_probability_legacy
    
    # Get operational ranges from probability analysis
    operational_ranges = plot_transport_probability_legacy(
        transport_modes, transport_modes_color, 
        probability_function, threshold
    )
    
    # Create monthly analysis
    df_modes_monthly = create_monthly_transport_analysis(
        G, operational_ranges, start_year, months_in_year, temperature_field
    )
    
    return df_modes_monthly, operational_ranges


# Fixed version of your original code
def create_transport_monthly_analysis_fixed(
    G_undirected: nx.Graph,
    transport_modes: List[str],
    operational_ranges: Dict[str, List[Tuple[float, float]]],
    START_YEAR: int,
    MONTHS_IN_YEAR: int = 12
) -> pd.DataFrame:
    """
    Fixed version of the original transport counting code
    
    Args:
        G_undirected: NetworkX graph with temperature data
        transport_modes: List of transport mode names
        operational_ranges: Operational ranges from plot_transport_probability_legacy
        START_YEAR: Starting year
        MONTHS_IN_YEAR: Number of months per year
        
    Returns:
        DataFrame with monthly transport mode analysis
    """
    # Extract time dimension
    sample_node = list(G_undirected.nodes(data=True))[0]
    T = sample_node[1]["temperature"].shape[0]
    n_years = T // MONTHS_IN_YEAR
    years = list(range(START_YEAR, START_YEAR + n_years))
    months = list(range(1, MONTHS_IN_YEAR + 1))
    
    records = []
    
    for t in range(T):
        year = START_YEAR + (t // MONTHS_IN_YEAR)
        month = (t % MONTHS_IN_YEAR) + 1
        
        # Count operational nodes for each mode
        mode_counts = {mode: 0 for mode in transport_modes}
        
        for node_id, data in G_undirected.nodes(data=True):
            # Safely extract scalar temperature value
            temp = extract_temperature_at_time(data["temperature"], t)
            
            for mode in transport_modes:
                if mode in operational_ranges:
                    ranges = operational_ranges[mode]
                    if is_temperature_in_operational_ranges(temp, ranges):
                        mode_counts[mode] += 1
        
        # Save records for each mode
        for mode in transport_modes:
            records.append({
                "year": year,
                "month": month,
                "month_name": pd.to_datetime(str(month), format="%m").strftime("%b"),
                "mode": mode,
                "count": mode_counts[mode],
            })
    
    return pd.DataFrame(records)


# Example usage function
def example_fixed_analysis():
    """Example of how to use the fixed transport analysis"""
    # # Step 1: Get operational ranges from probability analysis
    # operational_ranges = plot_transport_probability_legacy(
    #     transport_modes, transport_modes_color, 
    #     get_transport_probability, threshold
    # )
    # 
    # print("Operational ranges:")
    # for mode, ranges in operational_ranges.items():
    #     print(f"{mode}: {ranges}")
    # 
    # # Step 2: Create monthly analysis using the fixed function
    # df_modes_monthly = create_transport_monthly_analysis_fixed(
    #     G_undirected, transport_modes, operational_ranges, START_YEAR
    # )
    # 
    # # Step 3: Create heatmaps
    # create_transport_mode_heatmaps(
    #     df_modes_monthly=df_modes_monthly,
    #     transport_modes=transport_modes,
    #     month_order=month_order,
    #     max_nodes=len(G_undirected.nodes),
    #     invert_data=True  # Show operational periods
    # )
    pass


def debug_temperature_analysis(G: nx.Graph, operational_ranges: Dict[str, List[Tuple[float, float]]], 
                             time_index: int = 0) -> None:
    """
    Debug function to check temperature analysis
    
    Args:
        G: NetworkX graph
        operational_ranges: Operational ranges for transport modes
        time_index: Time index to analyze
    """
    print(f"Debug analysis for time index {time_index}:")
    print(f"Operational ranges: {operational_ranges}")
    print()
    
    # Sample a few nodes
    sample_nodes = list(G.nodes(data=True))[:5]
    
    for node_id, data in sample_nodes:
        temp = data["temperature"].values[time_index]
        print(f"Node {node_id}: Temperature = {temp:.1f}°C")
        
        for mode, ranges in operational_ranges.items():
            is_operational = is_temperature_in_operational_ranges(temp, ranges)
            print(f"  {mode}: {'OPERATIONAL' if is_operational else 'non-operational'}")
        print()
    
    # Count totals
    mode_counts = count_operational_nodes_by_mode(G, time_index, operational_ranges)
    print("Total operational nodes:")
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count} nodes")


# Legacy wrapper for backward compatibility
def create_df_modes_monthly_fixed(G_undirected, transport_modes, threshold_temperatures, START_YEAR, MONTHS_IN_YEAR=12):
    """
    Legacy wrapper that handles both old threshold format and new operational ranges format
    """
    print(f"Input threshold_temperatures: {threshold_temperatures}")
    
    # Check if threshold_temperatures is already in operational ranges format
    sample_value = next(iter(threshold_temperatures.values()))
    
    if isinstance(sample_value, list) and len(sample_value) > 0 and isinstance(sample_value[0], (tuple, list)):
        # Already in operational ranges format: {'mode': [(temp_from, temp_to), ...]}
        print("Detected operational ranges format - using directly")
        operational_ranges = {}
        
        for mode, ranges in threshold_temperatures.items():
            converted_ranges = []
            for range_tuple in ranges:
                if len(range_tuple) >= 2:
                    temp_from = float(range_tuple[0])
                    temp_to = float(range_tuple[1])
                    converted_ranges.append((temp_from, temp_to))
            operational_ranges[mode] = converted_ranges
            
    else:
        # Old format: {'mode': threshold_value} - convert to ranges
        print("Detected threshold values format - converting to ranges")
        operational_ranges = {}
        
        for mode, thresh in threshold_temperatures.items():
            # Ensure thresh is a scalar number
            if isinstance(thresh, (list, tuple)) and len(thresh) > 0:
                thresh = thresh[0]  # Take first element if it's a list
            elif hasattr(thresh, 'item'):
                thresh = thresh.item()  # Convert numpy scalar
            
            if not isinstance(thresh, (int, float)):
                print(f"Warning: Skipping {mode} - threshold is not a number: {thresh} (type: {type(thresh)})")
                continue
                
            if mode == "Winter road":
                # Winter road works below threshold
                operational_ranges[mode] = [(-100.0, float(thresh))]
            else:
                # Other modes work above threshold (this is oversimplified)
                operational_ranges[mode] = [(float(thresh), 100.0)]
    
    print(f"Final operational_ranges: {operational_ranges}")
    
    return create_transport_monthly_analysis_fixed(
        G_undirected, transport_modes, operational_ranges, START_YEAR, MONTHS_IN_YEAR
    )