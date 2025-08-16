import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Callable
from tqdm import tqdm
import copy


class Statistics:
    """Enhanced statistics tracking class for network analysis"""
    
    def __init__(self, custom_metrics: Optional[List[str]] = None):
        """
        Initialize statistics tracker
        
        Args:
            custom_metrics: List of additional metric names to track
        """
        base_columns = ["idx", "mean", "min", "max", "median"]
        if custom_metrics:
            base_columns.extend(custom_metrics)
            
        self.records = pd.DataFrame(columns=base_columns)
        self.graphs = []
        self.results = []
        self.custom_metrics = custom_metrics or []
    
    def add_record(self, step_index: int, metrics: Dict[str, float], 
                   G: nx.Graph, result: pd.DataFrame) -> None:
        """
        Add a new record to statistics
        
        Args:
            step_index: Index of the simulation step
            metrics: Dictionary of calculated metrics
            G: NetworkX graph for this step
            result: DataFrame with provision results
        """
        # Ensure all required metrics are present
        record_data = {
            "idx": step_index,
            "mean": metrics.get("mean", np.nan),
            "min": metrics.get("min", np.nan),
            "max": metrics.get("max", np.nan),
            "median": metrics.get("median", np.nan),
        }
        
        # Add custom metrics
        for metric in self.custom_metrics:
            record_data[metric] = metrics.get(metric, np.nan)
        
        new_row = pd.DataFrame([record_data])
        self.records = pd.concat([self.records, new_row], ignore_index=True)
        self.graphs.append(copy.deepcopy(G))
        self.results.append(result.copy())
    
    def summary(self, metric: str = "mean") -> pd.DataFrame:
        """
        Get summary statistics
        
        Args:
            metric: Metric to summarize
            
        Returns:
            DataFrame with summary statistics
        """
        if metric not in self.records.columns:
            raise ValueError(f"Metric '{metric}' not found in records")
        return self.records[["idx", metric]].set_index("idx")
    
    def full(self) -> pd.DataFrame:
        """Get full statistics records"""
        return self.records.copy()
    
    def get_time_series(self, metric: str) -> pd.Series:
        """
        Get time series for a specific metric
        
        Args:
            metric: Name of the metric
            
        Returns:
            Pandas Series with time series data
        """
        return self.records.set_index("idx")[metric]
    
    def get_statistics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        numeric_cols = self.records.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != "idx"]
        
        for col in numeric_cols:
            summary[col] = {
                "mean": self.records[col].mean(),
                "std": self.records[col].std(),
                "min": self.records[col].min(),
                "max": self.records[col].max(),
                "median": self.records[col].median()
            }
        
        return summary


class TemperatureHandler:
    """Handles temperature-related calculations for edges"""
    
    @staticmethod
    def get_edge_temperature(G: nx.Graph, u: Any, v: Any, idx: int, 
                           key: Any = None, temperature_field: str = "temperature") -> float:
        """
        Calculate edge temperature based on node temperatures and transport type
        
        Args:
            G: NetworkX graph
            u, v: Edge endpoints
            idx: Time index
            key: Edge key (for MultiGraph)
            temperature_field: Name of temperature field in node data
            
        Returns:
            Calculated edge temperature
        """
        temp_u = G.nodes[u].get(temperature_field).loc[idx].item()
        temp_v = G.nodes[v].get(temperature_field).loc[idx].item()
        
        # Get edge data
        if key is not None:
            edge_data = G.edges[u, v, key]
        else:
            edge_data = G.edges[u, v]
        
        transport_label = edge_data.get("label", "")
        
        # Apply temperature rule based on transport type
        if transport_label in ["car_cold", "winter_tr"]:
            return np.round(np.max([temp_u, temp_v]), 1)
        else:
            return np.round(np.min([temp_u, temp_v]), 1)
    
    @staticmethod
    def assign_edge_temperatures(G: nx.Graph, idx: int, 
                               temperature_field: str = "temperature") -> nx.Graph:
        """
        Assign temperatures to all edges in the graph
        
        Args:
            G: NetworkX graph
            idx: Time index
            temperature_field: Name of temperature field in node data
            
        Returns:
            Graph with temperature-assigned edges
        """
        G_copy = G.copy()
        
        if G.is_multigraph():
            for u, v, key in G_copy.edges(keys=True):
                G_copy.edges[u, v, key]["temperature"] = TemperatureHandler.get_edge_temperature(
                    G_copy, u, v, idx, key, temperature_field
                )
        else:
            for u, v in G_copy.edges():
                G_copy.edges[u, v]["temperature"] = TemperatureHandler.get_edge_temperature(
                    G_copy, u, v, idx, None, temperature_field
                )
        
        return G_copy


class TransportProbabilityCalculator:
    """Handles transport probability calculations"""
    
    def __init__(self, probability_function: Callable[[str, float], float]):
        """
        Initialize with a probability calculation function
        
        Args:
            probability_function: Function that takes (transport_type, temperature) and returns probability
        """
        self.probability_function = probability_function
    
    def assign_transport_probabilities(self, G: nx.Graph, 
                                     transport_field: str = "label",
                                     temperature_field: str = "temperature") -> nx.Graph:
        """
        Assign transport probabilities to all edges
        
        Args:
            G: NetworkX graph
            transport_field: Name of field containing transport type
            temperature_field: Name of field containing temperature
            
        Returns:
            Graph with probability-assigned edges
        """
        G_copy = G.copy()
        
        for u, v, data in G_copy.edges(data=True):
            transport_type = data.get(transport_field)
            temperature = data.get(temperature_field)
            
            if transport_type is not None and temperature is not None:
                data["transport_probability"] = self.probability_function(
                    transport_type, temperature
                )
            else:
                data["transport_probability"] = -1e3
        
        return G_copy


class GraphFilter:
    """Handles graph filtering operations"""
    
    @staticmethod
    def filter_edges_by_probability(G: nx.Graph, threshold: float, 
                                   probability_field: str = "transport_probability") -> nx.Graph:
        """
        Filter graph edges based on probability threshold
        
        Args:
            G: NetworkX graph
            threshold: Minimum probability threshold
            probability_field: Name of probability field
            
        Returns:
            Filtered graph
        """
        G_filtered = nx.Graph()
        G_filtered.add_nodes_from(G.nodes(data=True))
        
        for u, v, data in G.edges(data=True):
            prob = data.get(probability_field, 0.0)
            if prob >= threshold:
                G_filtered.add_edge(u, v, **data)
        
        # Preserve graph attributes
        if hasattr(G, 'graph'):
            G_filtered.graph.update(G.graph)
        
        return G_filtered
    
    @staticmethod
    def filter_by_custom_condition(G: nx.Graph, 
                                 condition_function: Callable[[Dict], bool]) -> nx.Graph:
        """
        Filter edges based on custom condition function
        
        Args:
            G: NetworkX graph
            condition_function: Function that takes edge data dict and returns bool
            
        Returns:
            Filtered graph
        """
        G_filtered = nx.Graph()
        G_filtered.add_nodes_from(G.nodes(data=True))
        
        for u, v, data in G.edges(data=True):
            if condition_function(data):
                G_filtered.add_edge(u, v, **data)
        
        # Preserve graph attributes
        if hasattr(G, 'graph'):
            G_filtered.graph.update(G.graph)
        
        return G_filtered


class MetricsCalculator:
    """Calculates various metrics from provision results"""
    
    @staticmethod
    def calculate_basic_metrics(result: pd.DataFrame, 
                              provision_field: str = "provision") -> Dict[str, float]:
        """
        Calculate basic provision metrics
        
        Args:
            result: DataFrame with provision data
            provision_field: Name of provision column
            
        Returns:
            Dictionary with calculated metrics
        """
        provision_series = result[provision_field]
        
        return {
            "mean": provision_series.mean().round(2),
            "min": provision_series.min().round(2),
            "max": provision_series.max().round(2),
            "median": provision_series.median().round(2),
            "std": provision_series.std().round(2),
            "q25": provision_series.quantile(0.25).round(2),
            "q75": provision_series.quantile(0.75).round(2)
        }
    
    @staticmethod
    def calculate_custom_metrics(result: pd.DataFrame, 
                               metric_functions: Dict[str, Callable]) -> Dict[str, float]:
        """
        Calculate custom metrics using provided functions
        
        Args:
            result: DataFrame with provision data
            metric_functions: Dictionary of {metric_name: function}
            
        Returns:
            Dictionary with calculated custom metrics
        """
        custom_metrics = {}
        for name, func in metric_functions.items():
            try:
                custom_metrics[name] = func(result)
            except Exception as e:
                print(f"Error calculating metric {name}: {e}")
                custom_metrics[name] = np.nan
        
        return custom_metrics


class AgglomerationNetwork:
    """Enhanced network analysis class with modular components"""
    
    def __init__(self, graph: nx.Graph, threshold: float,
                 probability_function: Callable[[str, float], float],
                 provision_calculator: Callable = None,
                 custom_metrics: Optional[List[str]] = None,
                 metric_functions: Optional[Dict[str, Callable]] = None):
        """
        Initialize the agglomeration network
        
        Args:
            graph: NetworkX graph
            threshold: Probability threshold for edge filtering
            probability_function: Function to calculate transport probabilities
            provision_calculator: Function to calculate provision (default: placeholder)
            custom_metrics: List of custom metric names to track
            metric_functions: Dictionary of custom metric calculation functions
        """
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.threshold = threshold
        self.current_step = 0
        
        # Initialize components
        self.stats = Statistics(custom_metrics)
        self.temperature_handler = TemperatureHandler()
        self.probability_calculator = TransportProbabilityCalculator(probability_function)
        self.graph_filter = GraphFilter()
        self.metrics_calculator = MetricsCalculator()
        
        # Store provision calculator
        self.provision_calculator = provision_calculator
        self.metric_functions = metric_functions or {}
    
    def run_single_step(self, idx: int, service_radius_minutes: int = 60,
                       base_demand: float = 1.0, service_name: str = "service",
                       return_assignment: bool = True) -> Tuple[nx.Graph, pd.DataFrame]:
        """
        Run a single simulation step
        
        Args:
            idx: Time step index
            service_radius_minutes: Service radius in minutes
            base_demand: Base demand value
            service_name: Name of the service
            return_assignment: Whether to return assignment matrix
            
        Returns:
            Tuple of (processed_graph, result_dataframe)
        """
        self.current_step = idx
        
        # Step 1: Assign edge temperatures
        graph_with_temp = self.temperature_handler.assign_edge_temperatures(
            self.graph, idx
        )
        
        # Step 2: Calculate transport probabilities
        graph_with_prob = self.probability_calculator.assign_transport_probabilities(
            graph_with_temp
        )
        
        # Step 3: Filter edges by probability threshold
        filtered_graph = self.graph_filter.filter_edges_by_probability(
            graph_with_prob, self.threshold
        )
        
        # Step 4: Calculate provision (using provided function or placeholder)
        if self.provision_calculator:
            if return_assignment:
                processed_graph, result, assignment = self.provision_calculator(
                    filtered_graph, service_radius_minutes, base_demand,
                    service_name=service_name, return_assignment=return_assignment
                )
            else:
                processed_graph, result = self.provision_calculator(
                    filtered_graph, service_radius_minutes, base_demand,
                    service_name=service_name, return_assignment=return_assignment
                )
        else:
            # Placeholder if no provision calculator provided
            processed_graph = filtered_graph
            result = pd.DataFrame({
                'name': [str(n) for n in filtered_graph.nodes()],
                'provision': np.random.rand(len(filtered_graph.nodes()))
            })
        
        # Step 5: Calculate metrics
        basic_metrics = self.metrics_calculator.calculate_basic_metrics(result)
        custom_metrics = self.metrics_calculator.calculate_custom_metrics(
            result, self.metric_functions
        )
        
        all_metrics = {**basic_metrics, **custom_metrics}
        
        # Step 6: Record statistics
        self.stats.add_record(idx, all_metrics, processed_graph, result)
        
        return processed_graph, result
    
    def run_all_steps(self, time_range: range, **step_kwargs) -> None:
        """
        Run simulation for all time steps
        
        Args:
            time_range: Range of time indices to simulate
            **step_kwargs: Additional arguments for run_single_step
        """
        for idx in tqdm(time_range, desc="Running network analysis"):
            self.run_single_step(idx, **step_kwargs)
    
    def get_statistics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all calculated statistics"""
        return self.stats.get_statistics_summary()
    
    def reset(self) -> None:
        """Reset the network to initial state"""
        self.graph = self.original_graph.copy()
        self.current_step = 0
        self.stats = Statistics(self.stats.custom_metrics)


# Factory function for easy setup
def create_agglomeration_network(graph: nx.Graph, threshold: float,
                               probability_function: Callable[[str, float], float],
                               provision_calculator: Callable = None,
                               custom_metrics: Optional[List[str]] = None) -> AgglomerationNetwork:
    """
    Factory function to create an AgglomerationNetwork with common configurations
    
    Args:
        graph: NetworkX graph
        threshold: Probability threshold
        probability_function: Transport probability function
        provision_calculator: Provision calculation function
        custom_metrics: Custom metrics to track
        
    Returns:
        Configured AgglomerationNetwork instance
    """
    return AgglomerationNetwork(
        graph=graph,
        threshold=threshold,
        probability_function=probability_function,
        provision_calculator=provision_calculator,
        custom_metrics=custom_metrics
    )


# Example usage and setup functions
def example_basic_setup():
    """Example of basic network setup"""
    # def get_transport_probability(transport_type, temperature):
    #     # Your probability calculation logic here
    #     return 0.5  # Placeholder
    # 
    # def calculate_provision(G, radius, demand, service_name="service", return_assignment=True):
    #     # Your provision calculation logic here
    #     result = pd.DataFrame({
    #         'name': [str(n) for n in G.nodes()],
    #         'provision': np.random.rand(len(G.nodes()))
    #     })
    #     if return_assignment:
    #         return G, result, np.eye(len(G.nodes()))
    #     return G, result
    # 
    # net = AgglomerationNetwork(
    #     graph=G_undirected,
    #     threshold=0.5,
    #     probability_function=get_transport_probability,
    #     provision_calculator=calculate_provision
    # )
    # 
    # # Run analysis
    # time_range = range(12)  # 12 months
    # net.run_all_steps(time_range, service_radius_minutes=60, base_demand=1.0)
    # 
    # # Get results
    # df_stats = net.stats.full()
    pass


def example_custom_metrics():
    """Example with custom metrics"""
    # def calculate_zero_provision_count(result):
    #     return (result['provision'] == 0).sum()
    # 
    # def calculate_high_provision_count(result):
    #     return (result['provision'] > 0.8).sum()
    # 
    # custom_metrics = ['zero_count', 'high_count']
    # metric_functions = {
    #     'zero_count': calculate_zero_provision_count,
    #     'high_count': calculate_high_provision_count
    # }
    # 
    # net = AgglomerationNetwork(
    #     graph=G_undirected,
    #     threshold=0.5,
    #     probability_function=get_transport_probability,
    #     provision_calculator=calculate_provision,
    #     custom_metrics=custom_metrics,
    #     metric_functions=metric_functions
    # )
    pass