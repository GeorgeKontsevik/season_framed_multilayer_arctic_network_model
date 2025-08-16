import plotly.graph_objects as go
from constants import month_order


def create_clean_sankey(
    graphs,
    month_start,
    service_name,
    min_flow=1,
):
    """
    Create clean Sankey: Consumers -> T1_Providers -> T2_Providers -> T3_Providers
    Simple and straightforward approach
    """

    print(f"Creating Sankey for {len(graphs)} time periods...")

    # Step 1: Get all unique consumers (from first graph), excluding self-sufficient ones
    G1 = graphs[0]
    consumers = {}  # {consumer_name: total_demand}

    # Find settlements with self-loops and full provision
    self_sufficient = set()
    for source, target, data in G1.edges(data=True):
        if data.get("is_service_flow", False) and source == target:  # Self-loop
            # Check if this settlement has full provision
            node_data = G1.nodes[source]
            provision = node_data.get("provision", 0)
            if provision >= 1.0:  # Fully provided
                self_sufficient.add(source)

    print(
        f"Excluding {len(self_sufficient)} self-sufficient settlements: {list(self_sufficient)}"
    )

    # Add consumers excluding self-sufficient ones
    for node_name, node_data in G1.nodes(data=True):
        demand = node_data.get("demand", 0)
        if demand > 0 and node_name not in self_sufficient:
            consumers[node_name] = demand

    print(f"Found {len(consumers)} consumers")

    # Step 2: Get provider assignments for each time period
    all_assignments = []  # [{consumer: provider}, {consumer: provider}, ...]

    for t, G in enumerate(graphs):
        assignments = {}

        # Get service flows (exclude self-service completely)
        for source, target, data in G.edges(data=True):
            if (
                data.get("is_service_flow", False)
                and data.get("assignment", 0) >= min_flow
                and source != target
            ):  # Only external service flows
                if source in consumers:  # Only consumers we know about
                    assignments[source] = target

        # For consumers with NO external assignments, they go to NO_PROVIDER
        # (This excludes self-sufficient settlements completely)
        for consumer in consumers:
            if consumer not in assignments:
                assignments[consumer] = "NO_PROVIDER"

        all_assignments.append(assignments)
        print(
            f"T{t+1}: {len([p for p in assignments.values() if p != 'NO_PROVIDER'])} assignments, "
            f"{len([p for p in assignments.values() if p == 'NO_PROVIDER'])} no provider"
        )

    # Step 3: Create Sankey nodes
    sankey_nodes = []
    node_indices = {}

    # Add consumers (sorted by demand)
    sorted_consumers = sorted(consumers.items(), key=lambda x: x[1], reverse=True)
    for consumer, demand in sorted_consumers:
        sankey_nodes.append(consumer)
        node_indices[consumer] = len(sankey_nodes) - 1

    # Add providers for each time period (sorted by total flow they receive)
    for t in range(len(graphs)):
        # Calculate total flow to each provider
        provider_flows = {}
        for consumer, provider in all_assignments[t].items():
            consumer_demand = consumers[consumer]
            if provider not in provider_flows:
                provider_flows[provider] = 0
            provider_flows[provider] += consumer_demand

        # Sort providers by flow
        sorted_providers = sorted(
            provider_flows.items(), key=lambda x: x[1], reverse=True
        )

        for provider, flow in sorted_providers:
            provider_label = f"{provider}_T{t+1}"
            sankey_nodes.append(provider_label)
            node_indices[provider_label] = len(sankey_nodes) - 1

    print(f"Created {len(sankey_nodes)} nodes")

    # Step 4: Create Sankey flows
    sankey_sources = []
    sankey_targets = []
    sankey_values = []

    # Flows from consumers to T1 providers
    for consumer, provider in all_assignments[0].items():
        consumer_demand = consumers[consumer]
        provider_label = f"{provider}_T1"

        source_idx = node_indices[consumer]
        target_idx = node_indices[provider_label]

        sankey_sources.append(source_idx)
        sankey_targets.append(target_idx)
        sankey_values.append(consumer_demand)

    # Flows between time periods (T1 -> T2, T2 -> T3, etc.)
    for t in range(len(all_assignments) - 1):
        current_assignments = all_assignments[t]
        next_assignments = all_assignments[t + 1]

        for consumer in consumers:
            consumer_demand = consumers[consumer]

            current_provider = current_assignments[consumer]
            next_provider = next_assignments[consumer]

            current_label = f"{current_provider}_T{t+1}"
            next_label = f"{next_provider}_T{t+2}"

            source_idx = node_indices[current_label]
            target_idx = node_indices[next_label]

            sankey_sources.append(source_idx)
            sankey_targets.append(target_idx)
            sankey_values.append(consumer_demand)

    print(f"Created {len(sankey_sources)} flows")

    # Step 6: Create colors
    node_colors = []
    for node in sankey_nodes:
        if node in consumers:  # Consumer
            node_colors.append("#1E88E5")  # Blue instead of red
        elif "NO_PROVIDER" in node:  # No provider
            node_colors.append("#FF4757")  # Dark red
        else:  # Provider
            node_colors.append("#26C6DA")  # Light teal

    # Create flow colors
    flow_colors = []
    for i in range(len(sankey_sources)):
        target_node = sankey_nodes[sankey_targets[i]]
        if "NO_PROVIDER" in target_node:
            flow_colors.append("rgba(255, 71, 87, 0.7)")  # Red for NO_PROVIDER
        else:
            flow_colors.append("rgba(149, 165, 166, 0.5)")  # Gray for normal flows

    # Step 7: Create Plotly figure with simpler positioning
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.8),
                    label=sankey_nodes,
                    color=node_colors,
                    # Remove x,y positioning to let Plotly handle it naturally
                ),
                link=dict(
                    source=sankey_sources,
                    target=sankey_targets,
                    value=sankey_values,
                    color=flow_colors,
                ),
            )
        ]
    )

    # Determine number of time periods for dynamic headers
    num_periods = len(graphs)

    # Create column headers based on number of periods
    annotations = [
        dict(
            x=0.00,
            y=1.05,
            text="<b>Consumers</b>",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=14),
        ),
    ]

    # Add provider column headers dynamically
    for t in range(num_periods):
        x_pos = 0.16 + ((t) / (num_periods))  # Spread provider columns evenly
        annotations.append(
            dict(
                x=x_pos,
                y=1.05,
                text=f"<b>{month_order[month_start+t]}</b>",
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(size=14),
            )
        )

    fig.update_layout(
        title_text=f"{service_name.title()} Service Flow Redistribution",
        font_size=11,
        width=1400,
        height=700,
        annotations=annotations,
    )

    fig.show()
    return fig
