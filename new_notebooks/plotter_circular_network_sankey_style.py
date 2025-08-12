import plotly.graph_objects as go
import numpy as np
import math


def plot_circular_network_sankey_style(g, service_name, month_name, min_flow=1):
    """
    Create a circular network plot that matches the Sankey data processing exactly
    Cities are represented as flat segments along the circle perimeter

    Parameters:
    g: Single NetworkX graph
    service_name: Name of the service
    min_flow: Minimum flow threshold
    """

    # print(f"Creating circular network...")

    # Step 1: Get all unique consumers, excluding self-sufficient ones
    consumers = {}  # {consumer_name: demand}

    # Find settlements with self-loops and full provision
    self_sufficient = set()
    for source, target, data in g.edges(data=True):
        if data.get("is_service_flow", False) and source == target:  # Self-loop
            # Check if this settlement has full provision
            node_data = g.nodes[source]
            provision = node_data.get("provision", 0)
            if provision >= 1.0:  # Fully provided
                self_sufficient.add(source)

    # print(
    #     f"Excluding {len(self_sufficient)} self-sufficient settlements: {list(self_sufficient)}"
    # )

    # Add consumers excluding self-sufficient ones
    for node_name, node_data in g.nodes(data=True):
        demand = node_data.get("demand", 0)
        if demand > 0 and node_name not in self_sufficient:
            consumers[node_name] = demand

    # print(f"Found {len(consumers)} consumers")

    # Step 2: Get provider assignments
    assignments = {}

    # Get service flows (exclude self-service completely)
    for source, target, data in g.edges(data=True):
        if (
            data.get("is_service_flow", False)
            and data.get("assignment", 0) >= min_flow
            and source != target
        ):  # Only external service flows
            if source in consumers:  # Only consumers we know about
                assignments[source] = target

    # For consumers with NO external assignments, they go to NO_PROVIDER
    for consumer in consumers:
        if consumer not in assignments:
            assignments[consumer] = "NO_PROVIDER"

    # print(
    #     f"{len([p for p in assignments.values() if p != 'NO_PROVIDER'])} assignments, "
    #     f"{len([p for p in assignments.values() if p == 'NO_PROVIDER'])} no provider"
    # )

    # Step 3: Create network data
    providers = set(assignments.values())
    all_nodes = list(consumers.keys()) + [p for p in providers if p != "NO_PROVIDER"]

    # Calculate provider flows for sizing
    provider_flows = {}
    for consumer, provider in assignments.items():
        consumer_demand = consumers[consumer]
        if provider not in provider_flows:
            provider_flows[provider] = 0
        provider_flows[provider] += consumer_demand

    # Step 4: Create circular layout with segments
    fig = go.Figure()

    # Sort nodes: consumers first (by demand), then providers (by flow), then NO_PROVIDER
    sorted_consumers = sorted(consumers.items(), key=lambda x: x[1], reverse=True)
    sorted_providers = sorted(
        [(p, f) for p, f in provider_flows.items() if p != "NO_PROVIDER"],
        key=lambda x: x[1],
        reverse=True,
    )

    # Add NO_PROVIDER if it exists
    ordered_nodes = [c[0] for c in sorted_consumers] + [p[0] for p in sorted_providers]
    if "NO_PROVIDER" in provider_flows:
        ordered_nodes.append("NO_PROVIDER")

    # Calculate total demand for proportional segments
    total_demand = sum(consumers.values()) + sum(f for p, f in sorted_providers)
    if "NO_PROVIDER" in provider_flows:
        total_demand += provider_flows["NO_PROVIDER"]

    # print(f"Total demand: {total_demand}")
    # print(f"Ordered nodes: {ordered_nodes}")

    # Create segment positions
    segment_positions = {}
    current_angle = 0
    radius = 1.0

    # Calculate total space available (full circle minus gaps)
    num_segments = len(ordered_nodes)
    gap_size = 0.01
    total_gap_space = gap_size * num_segments
    available_space = 2 * math.pi - total_gap_space

    for node_id in ordered_nodes:
        if node_id in consumers:
            demand = consumers[node_id]
        elif node_id == "NO_PROVIDER":
            demand = provider_flows["NO_PROVIDER"]
        else:
            demand = provider_flows[node_id]

        # Calculate segment size proportional to demand (using available space)
        segment_size = max(0.01, (demand / total_demand) * available_space)

        start_angle = current_angle
        end_angle = current_angle + segment_size
        mid_angle = (start_angle + end_angle) / 2

        segment_positions[node_id] = {
            "start_angle": start_angle,
            "end_angle": end_angle,
            "mid_angle": mid_angle,
            "mid_x": math.cos(mid_angle) * radius,
            "mid_y": math.sin(mid_angle) * radius,
        }

        current_angle = end_angle + gap_size  # Add gap after each segment

        # print(
        #     f"{node_id}: demand={demand}, segment_size={segment_size:.3f}, angles={start_angle:.3f}-{end_angle:.3f}"
        # )

    # Step 5: Draw city segments
    # Colors matching Sankey diagram
    consumer_color = "#1E88E5"  # Blue for consumers
    provider_color = "#26C6DA"  # Light teal for providers
    no_provider_color = "#FF4757"  # Dark red for NO_PROVIDER

    # Track legend groups to avoid duplicates
    shown_legend_groups = set()

    # Draw consumer segments
    for consumer, demand in sorted_consumers:
        pos = segment_positions[consumer]

        # Create arc for segment
        angles = np.linspace(pos["start_angle"], pos["end_angle"], 20)
        inner_radius = 0.9
        outer_radius = 1.0

        # Create segment shape
        x_outer = [math.cos(a) * outer_radius for a in angles]
        y_outer = [math.sin(a) * outer_radius for a in angles]
        x_inner = [math.cos(a) * inner_radius for a in reversed(angles)]
        y_inner = [math.sin(a) * inner_radius for a in reversed(angles)]

        x_segment = x_outer + x_inner + [x_outer[0]]
        y_segment = y_outer + y_inner + [y_outer[0]]

        provider = assignments[consumer]

        show_legend = "consumers" not in shown_legend_groups
        if show_legend:
            shown_legend_groups.add("consumers")

        fig.add_trace(
            go.Scatter(
                x=x_segment,
                y=y_segment,
                fill="toself",
                mode="none",
                fillcolor=consumer_color,
                name="Consumers",
                legendgroup="consumers",
                showlegend=show_legend,
                hovertemplate=f"<b>{consumer}</b><br>Demand: {demand:.1f}<br>Provider: {provider}<extra></extra>",
            )
        )

        # Add text label
        fig.add_annotation(
            x=pos["mid_x"] * 1.15,
            y=pos["mid_y"] * 1.15,
            text=consumer,
            showarrow=False,
            font=dict(size=10, color="#2c3e50"),
            xanchor="center",
            yanchor="middle",
        )

    # Draw provider segments
    for provider, flow in sorted_providers:
        pos = segment_positions[provider]

        # Create arc for segment
        angles = np.linspace(pos["start_angle"], pos["end_angle"], 20)
        inner_radius = 0.9
        outer_radius = 1.0

        # Create segment shape
        x_outer = [math.cos(a) * outer_radius for a in angles]
        y_outer = [math.sin(a) * outer_radius for a in angles]
        x_inner = [math.cos(a) * inner_radius for a in reversed(angles)]
        y_inner = [math.sin(a) * inner_radius for a in reversed(angles)]

        x_segment = x_outer + x_inner + [x_outer[0]]
        y_segment = y_outer + y_inner + [y_outer[0]]

        served_consumers = [c for c, p in assignments.items() if p == provider]

        show_legend = "providers" not in shown_legend_groups
        if show_legend:
            shown_legend_groups.add("providers")

        fig.add_trace(
            go.Scatter(
                x=x_segment,
                y=y_segment,
                fill="toself",
                mode="none",
                fillcolor=provider_color,
                name="Providers",
                legendgroup="providers",
                showlegend=show_legend,
                hovertemplate=f"<b>{provider}</b><br>Total Flow: {flow:.1f}<br>Serves: {len(served_consumers)} consumers<extra></extra>",
            )
        )

        # Add text label
        fig.add_annotation(
            x=pos["mid_x"] * 1.15,
            y=pos["mid_y"] * 1.15,
            text=provider,
            showarrow=False,
            font=dict(size=10, color="#2c3e50"),
            xanchor="center",
            yanchor="middle",
        )

    # Draw NO_PROVIDER segment if it exists
    if "NO_PROVIDER" in provider_flows:
        pos = segment_positions["NO_PROVIDER"]
        flow = provider_flows["NO_PROVIDER"]

        # Create arc for segment
        angles = np.linspace(pos["start_angle"], pos["end_angle"], 20)
        inner_radius = 0.9
        outer_radius = 1.0

        # Create segment shape
        x_outer = [math.cos(a) * outer_radius for a in angles]
        y_outer = [math.sin(a) * outer_radius for a in angles]
        x_inner = [math.cos(a) * inner_radius for a in reversed(angles)]
        y_inner = [math.sin(a) * inner_radius for a in reversed(angles)]

        x_segment = x_outer + x_inner + [x_outer[0]]
        y_segment = y_outer + y_inner + [y_outer[0]]

        unserved_consumers = [c for c, p in assignments.items() if p == "NO_PROVIDER"]

        fig.add_trace(
            go.Scatter(
                x=x_segment,
                y=y_segment,
                fill="toself",
                mode="none",
                fillcolor=no_provider_color,
                name="No Provider",
                legendgroup="no_provider",
                showlegend=True,
                hovertemplate=f"<b>NO PROVIDER</b><br>Total Flow: {flow:.1f}<br>Unserved: {len(unserved_consumers)} consumers<extra></extra>",
            )
        )

        # Add text label
        fig.add_annotation(
            x=pos["mid_x"] * 1.15,
            y=pos["mid_y"] * 1.15,
            text="NO PROVIDER",
            showarrow=False,
            font=dict(size=10, color="white", family="Arial Black"),
            bgcolor="rgba(255, 71, 87, 0.8)",
            bordercolor="white",
            borderwidth=1,
            xanchor="center",
            yanchor="middle",
        )

    # Step 6: Draw connections (curved lines from segment to segment)
    # Track transport modes for legend
    transport_mode_traces = {}

    transport_modes_color = {
        "Aviation": "#8E44AD",  # Purple
        "Regular road": "#F39C12",  # Orange
        "Winter road": "#27AE60",  # Green
        "Water transport": "#3498DB",  # Blue
        "Service Connection": "#95A5A6",  # Gray for generic service flows
        "Unknown": "#BDC3C7",  # Light gray
    }

    # First, draw flows from consumers to providers (excluding NO_PROVIDER)
    for consumer, provider in assignments.items():
        if provider == "NO_PROVIDER":
            continue

        consumer_pos = segment_positions[consumer]
        provider_pos = segment_positions[provider]

        x0, y0 = consumer_pos["mid_x"] * 0.85, consumer_pos["mid_y"] * 0.85
        x1, y1 = provider_pos["mid_x"] * 0.85, provider_pos["mid_y"] * 0.85

        consumer_demand = consumers[consumer]

        # Get the transport mode from assignment edge
        edge_label = "Unknown"
        for source, target, data in g.edges(data=True):
            if (
                source == consumer
                and target == provider
                and data.get("is_service_flow", False)
            ):
                # For service flows, check if there's a transport mode in the assignment
                assignment_data = data.get("assignment", 0)
                if assignment_data > 0:
                    # Look for transport mode in edge data
                    if "transport_mode" in data:
                        edge_label = data["transport_mode"]
                    elif "mode" in data:
                        edge_label = data["mode"]
                    elif "label" in data:
                        edge_label = data["label"]
                    else:
                        # If no transport mode specified, it's just a service flow
                        edge_label = "Multi-path"
                break

        # Create curved path toward center
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        control_x = mid_x * 0.3  # Curve toward center
        control_y = mid_y * 0.3

        # Create smooth curve
        t = np.linspace(0, 1, 50)
        curve_x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * control_x + t**2 * x1
        curve_y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * control_y + t**2 * y1

        # Scale thickness based on demand
        line_width = max(1, min(8, consumer_demand / 15))

        # Get color for this edge label
        color = transport_modes_color.get(edge_label, transport_modes_color["Unknown"])

        # Track whether this transport mode has been added to legend
        show_legend = edge_label not in transport_mode_traces
        if show_legend:
            transport_mode_traces[edge_label] = True

        fig.add_trace(
            go.Scatter(
                x=curve_x,
                y=curve_y,
                mode="lines",
                name=edge_label,
                legendgroup=f"transport_{edge_label}",
                line=dict(
                    color=color,
                    width=line_width,
                    shape="spline",
                    smoothing=1.0,
                ),
                hovertemplate=f"<b>{consumer} → {provider}</b><br>Type: {edge_label}<br>Demand: {consumer_demand:.1f}<extra></extra>",
                showlegend=show_legend,
            )
        )

    # Second, draw NO_PROVIDER flows (red lines to center or to NO_PROVIDER segment)
    no_provider_consumers = [c for c, p in assignments.items() if p == "NO_PROVIDER"]
    if no_provider_consumers and "NO_PROVIDER" in segment_positions:
        no_provider_pos = segment_positions["NO_PROVIDER"]

        # Show legend only once for NO_PROVIDER flows
        show_no_provider_legend = True

        for consumer in no_provider_consumers:
            consumer_pos = segment_positions[consumer]
            consumer_demand = consumers[consumer]

            x0, y0 = consumer_pos["mid_x"] * 0.85, consumer_pos["mid_y"] * 0.85
            x1, y1 = no_provider_pos["mid_x"] * 0.85, no_provider_pos["mid_y"] * 0.85

            # Create curved path toward NO_PROVIDER segment
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            control_x = mid_x * 0.2  # Less curve for NO_PROVIDER flows
            control_y = mid_y * 0.2

            # Create smooth curve
            t = np.linspace(0, 1, 50)
            curve_x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * control_x + t**2 * x1
            curve_y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * control_y + t**2 * y1

            # Scale thickness based on demand
            line_width = max(2, min(8, consumer_demand / 15))

            fig.add_trace(
                go.Scatter(
                    x=curve_x,
                    y=curve_y,
                    mode="lines",
                    name="No Provider Flow",
                    legendgroup="no_provider_flow",
                    line=dict(
                        color=no_provider_color,  # Red color
                        width=line_width,
                        shape="spline",
                        smoothing=1.0,
                    ),
                    hovertemplate=f"<b>{consumer} → NO PROVIDER</b><br>Unmet Demand: {consumer_demand:.1f}<extra></extra>",
                    showlegend=show_no_provider_legend,
                )
            )

            # Only show legend for first NO_PROVIDER flow
            show_no_provider_legend = False

    # Step 7: Update layout
    fig.update_layout(
        title=dict(
            text=f"{month_name} | {service_name.title()}",
            font=dict(size=18, family="Arial", color="#2c3e50"),
            x=0.5,
            xanchor="center",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        hovermode="closest",
        margin=dict(b=80, l=80, r=80, t=80),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.4, 1.4],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]
        ),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12, color="#2c3e50"),
        width=1000,
        height=1000,
    )

    return fig

