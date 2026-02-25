import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle
import matplotlib.patches as mpatches
from collections import defaultdict
from scripts.preprocesser.constants import SERVICE_COLORS, service_list, month_order
from scripts.calculator.calculator_metrics import identify_stable_communities, calculate_temporal_metrics, create_temporal_summary_report

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from collections import defaultdict
import seaborn as sns
from scipy.stats import entropy
import networkx as nx

# Usage:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from collections import defaultdict
import seaborn as sns
from scipy.stats import entropy
import networkx as nx


# Publication-ready settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.5,
})

from collections import defaultdict


def run_complete_temporal_analysis(all_results, settl_name, month_range=range(4, 10)):
    """Main runner for complete temporal analysis"""
    print(f"🔄 Running analysis: {settl_name}")
    print(f"   Services: {len(service_list)} | Months: {month_order[month_range.start]}-{month_order[month_range.stop-1]}")
    print("="*60)
    
    # 1. Temporal evolution
    print("\n📅 Temporal evolution...")
    plot_temporal_service_evolution(all_results, settl_name, month_range)
    plt.show()
    
    # 2. Stable communities
    # print("\n🏛️ Stable communities...")
    # multi, temporal, super_stable = plot_stable_communities(all_results, settl_name, month_range)
    # plt.show()
    
    # print(f"   Multi-service: {len(multi)} | Temporal: {len(temporal)} | Super: {len(super_stable)}")
    
    # if super_stable:
    #     top3 = sorted(super_stable.items(), key=lambda x: x[1]['stability_score'], reverse=True)[:5]
    #     for i, (p, d) in enumerate(top3, 1):
    #         print(f"   #{i} {p}: {len(d['services'])} services, score={d['stability_score']:.2f}")
    
    # 3. Metrics
    print("\n📊 Temporal metrics...")
    metrics, communities = calculate_temporal_metrics(all_results, settl_name, month_range)
    plot_temporal_metrics(metrics, communities)
    plt.show()
    
    # 4. Report
    print(create_temporal_summary_report(metrics, communities))
    
    # return {
    #     'metrics': metrics,
    #     'communities': communities,
    #     'multi_service': multi,
    #     'temporal_stable': temporal,
    #     'super_stable': super_stable
    # }

def quick_single_month_analysis(all_results, settl_name, month_idx=5):
    """Quick single month visualization"""
    print(f"\n📍 Month: {month_order[month_idx]}")
    
    pmap, provs = plot_enhanced_service_areas(all_results, settl_name, month_idx=month_idx)
    plt.show()
    
    print(f"   Providers: {sum(len(p) for p in provs.values())} | Services: {len(provs)}")
    return pmap, provs


def plot_temporal_metrics(temporal_metrics, figsize=(18, 5)):
    """
    Publication-quality visualization of temporal metrics with enhanced styling
    """
    # Enhanced color palette - more vibrant and modern
    colors = {
        'primary': '#0077BE',      # Bright Blue
        'secondary': '#8E44AD',    # Rich Purple
        'accent': '#E67E22',       # Vibrant Orange
        'success': '#E74C3C',      # Bold Red
        'info': '#9B59B6',         # Deep purple
        'warning': '#F39C12',      # Golden Yellow
        'neutral': '#34495E'       # Dark Gray
    }
    
    # Create figure with enhanced styling
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Enhanced global font properties
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'font.weight': 'normal',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 1.0,
        'grid.linestyle': '--',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral']
    })
    
    # Create subplot grid with enhanced spacing
    gs = fig.add_gridspec(1, 3, hspace=0.4, wspace=0.35, 
                         left=0.06, right=0.96, top=0.88, bottom=0.15)
    
    # 1. Enhanced Jaccard Similarity Timeline
    ax1 = fig.add_subplot(gs[0, 0])
    if temporal_metrics['jaccard_similarity']:
        transitions = list(temporal_metrics['jaccard_similarity'].keys())
        values = list(temporal_metrics['jaccard_similarity'].values())
        
        # Create enhanced line plot with shadow effect
        ax1.plot(range(len(transitions)), values, 'o-', 
                color=colors['primary'], linewidth=3.5, markersize=9,
                markerfacecolor='white', markeredgewidth=2.5, 
                markeredgecolor=colors['primary'], alpha=0.9,
                markerfacecoloralt=colors['primary'])
        
        # Add subtle shadow line
        ax1.plot(range(len(transitions)), values, 'o-', 
                color=colors['primary'], linewidth=1.5, markersize=9,
                alpha=0.3, zorder=0)
        
        ax1.set_xticks(range(len(transitions)))
        ax1.set_xticklabels(transitions, rotation=45, ha='right', fontsize=11, 
                           color=colors['neutral'], fontweight='medium')
        ax1.set_ylabel('Jaccard Similarity', fontsize=13, fontweight='bold', 
                      color=colors['neutral'])
        ax1.set_title('Community Similarity\nBetween Consecutive Months', 
                     fontsize=14, fontweight='bold', pad=20, color=colors['neutral'])
        ax1.set_ylim([0, 1.05])
        ax1.grid(True, alpha=0.25, linestyle='--', linewidth=1.0)
        
        # Enhanced background with subtle gradient effect
        ax1.set_facecolor('#F8F9FA')
        
        # Add subtle border
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("#FFFFFF")
    
    # 2. Enhanced NMI Scores
    ax2 = fig.add_subplot(gs[0, 1])
    if temporal_metrics['nmi_scores']:
        transitions = list(temporal_metrics['nmi_scores'].keys())
        values = list(temporal_metrics['nmi_scores'].values())
        
        # Create enhanced line plot with different marker style
        ax2.plot(range(len(transitions)), values, 's-', 
                color=colors['secondary'], linewidth=3.5, markersize=8,
                markerfacecolor='white', markeredgewidth=2.5, 
                markeredgecolor=colors['secondary'], alpha=0.9)
        
        # Add subtle shadow line
        ax2.plot(range(len(transitions)), values, 's-', 
                color=colors['secondary'], linewidth=1.5, markersize=8,
                alpha=0.3, zorder=0)
        
        ax2.set_xticks(range(len(transitions)))
        ax2.set_xticklabels(transitions, rotation=45, ha='right', fontsize=11,
                           color=colors['neutral'], fontweight='medium')
        ax2.set_ylabel('NMI Score', fontsize=13, fontweight='bold',
                      color=colors['neutral'])
        ax2.set_title('Normalized Mutual\nInformation', 
                     fontsize=14, fontweight='bold', pad=20, color=colors['neutral'])
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.25, linestyle='--', linewidth=1.0)
        ax2.set_facecolor('#F8F9FA')
        
        # Add subtle border
        for spine in ax2.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#E0E0E0')
    
    # 3. Enhanced Community Evolution Stacked Bar
    ax3 = fig.add_subplot(gs[0, 2])
    if temporal_metrics['community_evolution']:
        evolution_data = defaultdict(list)
        transitions = list(temporal_metrics['community_evolution'].keys())

        # Enhanced color palette for stacked bars with better contrast
        stack_colors = {
            'Persistence': '#3498DB',      # Bright Blue
            'Growth': '#2ECC71',       # Emerald Green  
            'Contraction': '#E67E22',      # Vibrant Orange
            'Split': '#E74C3C',       # Bold Red
            'Merge': '#9B59B6',      # Amethyst Purple
            'Dissolution': '#95A5A6', # Light Gray
            'Birth': '#F39C12'          # Golden Yellow
        }
        
        for transition in transitions:
            for event_type in list(stack_colors.keys()):
                evolution_data[event_type].append(
                    temporal_metrics['community_evolution'][transition].get(event_type, 0)
                )
        
        bottom = np.zeros(len(transitions))
        
        
        
        bars = []
        for event_type in list(stack_colors.keys()):
            if event_type in evolution_data:
                values = evolution_data[event_type]
                bar = ax3.bar(range(len(transitions)), values, bottom=bottom, 
                             label=event_type.capitalize(), color=stack_colors[event_type], 
                             alpha=0.9, edgecolor='white', linewidth=1.2,
                             width=0.8)
                bars.append(bar)
                bottom += np.array(values)
        
        ax3.set_xticks(range(len(transitions)))
        ax3.set_xticklabels(transitions, rotation=45, ha='right', fontsize=11,
                           color=colors['neutral'], fontweight='medium')
        ax3.set_ylabel('Number of Communities', fontsize=13, fontweight='bold',
                      color=colors['neutral'])
        ax3.set_title('Community Evolution Events', 
                     fontsize=14, fontweight='bold', pad=20, color=colors['neutral'])
        
        # Enhanced legend with better positioning
        ax3.legend(loc='center', fontsize=10, ncol=4, 
                  bbox_to_anchor=(0.5, -.3), framealpha=0.9)
        
        ax3.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=1.0)
        ax3.set_facecolor('#F8F9FA')
        
        # Add subtle border
        for spine in ax3.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#E0E0E0')

    # Enhanced figure-wide styling
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    
    # Add a subtle title for the entire figure
    # fig.suptitle('Temporal Community Analysis', fontsize=16, fontweight='bold', 
    #             y=0.95, color=colors['neutral'])
    
    plt.tight_layout()
    return fig



def plot_enhanced_service_areas(all_results, settl_name, month_idx, 
                                figsize=(16, 12), show_provider_halos=True):
    """
    Enhanced visualization with provider halos and better distinction
    """
    # Use global defaults if not provided

    # Get positions and provider-consumer relationships
    pos = {}
    provider_consumer_map = defaultdict(lambda: defaultdict(set))
    providers_by_service = defaultdict(set)
    
    for service in service_list:
        try:
            graph = all_results[settl_name][service]["stats"].graphs[month_idx]
            
            # Get positions
            if not pos:
                for node, data in graph.nodes(data=True):
                    if "x" in data and "y" in data:
                        pos[node] = (data["x"], data["y"])
                    elif "longitude" in data and "latitude" in data:
                        pos[node] = (data["longitude"], data["latitude"])
            
            # Extract provider-consumer relationships
            for source, target, data in graph.edges(data=True):
                if (data.get("is_service_flow", False) and 
                    data.get("assignment", 0) > 0 and 
                    source != target):
                    provider_consumer_map[service][target].add(source)
                    providers_by_service[service].add(target)
                    
        except Exception as e:
            continue
    
    if not pos:
        print("No position data found!")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot base nodes
    all_x = [pos[node][0] for node in pos]
    all_y = [pos[node][1] for node in pos]
    ax.scatter(all_x, all_y, c='lightgray', s=15, alpha=0.4, zorder=1)
    
    # Plot service areas with provider halos
    legend_elements = []
    provider_count = 0
    
    for service_idx, service in enumerate(service_list):
        if service not in provider_consumer_map:
            continue
            
        color = SERVICE_COLORS.get(service, "#34495e")
        
        for provider_idx, (provider, consumers) in enumerate(provider_consumer_map[service].items()):
            if len(consumers) >= 1:
                group_nodes = consumers | {provider}
                group_pos = [pos[node] for node in group_nodes if node in pos]
                
                # Draw convex hull for consumer area
                if len(group_pos) >= 3:
                    try:
                        points = np.array(group_pos)
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        
                        polygon = Polygon(hull_points, alpha=0.15, 
                                        facecolor=color, edgecolor=color,
                                        linewidth=1.5, zorder=2 + service_idx * 0.1)
                        ax.add_patch(polygon)
                    except:
                        pass
                
                # Draw provider halo if enabled
                if show_provider_halos and provider in pos:
                    # Multiple concentric circles for provider
                    for radius_mult in [3.0, 2.0, 1.0]:
                        halo_radius = 0.0002 * radius_mult  # Adjust based on your coordinate scale
                        halo = Circle(pos[provider], halo_radius, 
                                    alpha=0.1 * radius_mult, 
                                    facecolor=color, 
                                    edgecolor=color,
                                    linewidth=0.5,
                                    zorder=5 + radius_mult)
                        ax.add_patch(halo)
                
                # Plot consumer nodes
                consumer_x = [pos[node][0] for node in consumers if node in pos]
                consumer_y = [pos[node][1] for node in consumers if node in pos]
                
                ax.scatter(consumer_x, consumer_y, c=color, s=30, alpha=0.7,
                          edgecolors='white', linewidth=0.5, zorder=10)
                
                # Highlight provider with distinct marker
                if provider in pos:
                    ax.scatter(pos[provider][0], pos[provider][1], 
                              c=color, s=100, marker='*',
                              edgecolors='black', linewidth=1.5, zorder=12,
                              label=f'{service} provider' if provider_idx == 0 else "")
                    provider_count += 1
        
        # Add to legend
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, markersize=8, 
                                         label=service))
    
    month_name = month_order[month_idx] if month_idx < len(month_order) else f"Month {month_idx}"
    ax.set_title(f'Service Areas with Provider Halos - {settl_name} ({month_name})\n'
                 f'Total Providers: {provider_count}', fontsize=14)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    
    return provider_consumer_map, providers_by_service

def plot_temporal_service_evolution(all_results, settl_name, 
                                   month_range, figsize=(10, 5)):
    """
    Visualize service areas evolution across all months in the range
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    
    months = list(month_range)
    n_months = len(months)
    
    # Create subplots grid with extra space for legend
    cols = 2
    rows = (n_months + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_months > 1 else [axes]
    
    # Get consistent positions from first available month
    all_positions = {}
    for month_idx in months:
        for service in service_list:
            try:
                graph = all_results[settl_name][service]["stats"].graphs[month_idx]
                for node, data in graph.nodes(data=True):
                    if node not in all_positions:
                        if "x" in data and "y" in data:
                            all_positions[node] = (data["x"], data["y"])
                        elif "longitude" in data and "latitude" in data:
                            all_positions[node] = (data["longitude"], data["latitude"])
                if all_positions:
                    break
            except:
                continue
        if all_positions:
            break
    
    # Collect all services that appear across all months for legend
    all_services_used = set()
    
    # Plot for each month
    for plot_idx, month_idx in enumerate(months):
        ax = axes[plot_idx]
        
        # Base nodes
        if all_positions:
            all_x = [all_positions[node][0] for node in all_positions]
            all_y = [all_positions[node][1] for node in all_positions]
            ax.scatter(all_x, all_y, c='lightgray', s=100, alpha=0.7, zorder=1, label='Base nodes', edgecolors='white', linewidth=0.5)
        
        provider_count = 0
        service_coverage = set()
        
        # Plot each service
        for c, service in enumerate(service_list):
            try:
                graph = all_results[settl_name][service]["stats"].graphs[month_idx]
                color = SERVICE_COLORS.get(service, "#34495e")
                
                # Extract provider-consumer relationships
                providers = defaultdict(set)
                for source, target, data in graph.edges(data=True):
                    if (data.get("is_service_flow", False) and 
                        data.get("assignment", 0) > 0 and 
                        source != target):
                        providers[target].add(source)
                        service_coverage.add(service)
                
                # Draw areas for each provider
                service_has_areas = False
                for provider, consumers in providers.items():
                    if len(consumers) >= 1:
                        group_nodes = consumers | {provider}
                        group_pos = [all_positions[node] for node in group_nodes if node in all_positions]
                        
                        if len(group_pos) >= 2:
                            try:
                                points = np.array(group_pos)
                                hull = ConvexHull(points)
                                hull_points = points[hull.vertices]
                                
                                polygon = Polygon(hull_points, alpha=0.4, 
                                                facecolor=color, edgecolor=color,
                                                linewidth=1, zorder=2)
                                ax.add_patch(polygon)
                                service_has_areas = True
                            except:
                                pass
                        
                        # Mark provider
                        if provider in all_positions:
                            ax.scatter(all_positions[provider][0], all_positions[provider][1], 
                                     c=color, s=300/c, marker='o', 
                                     edgecolors='white', linewidth=1,)
                            provider_count += 1
                
                # Track services that actually have coverage areas
                if service_has_areas:
                    all_services_used.add(service)
                    
            except:
                continue
        
        month_name = month_order[month_idx] if month_idx < len(month_order) else f"Month {month_idx}"
        ax.set_title(f'{month_name}\nProviders: {provider_count}, Services: {len(service_coverage)}', 
                    fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_months, len(axes)):
        axes[idx].axis('off')
    
    # Create publication-quality legend at bottom
    legend_elements = []
    
    # Node legend elements
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                                 markeredgecolor='gray', markeredgewidth=0.3,
                                 markersize=8, alpha=0.8, label='Network nodes'))
    
    # Service provider and area legend elements (grouped by service)
    for service in sorted(all_services_used):
        color = SERVICE_COLORS.get(service, "#34495e")
        
        # Provider (star)
        legend_elements.append(Line2D([0], [0], marker='*', color='w', 
                                     markerfacecolor=color, markeredgecolor='white', 
                                     markeredgewidth=0.8, markersize=12, 
                                     label=f'{service.title()} provider'))
        
        # Coverage area (polygon)
        legend_elements.append(mpatches.Patch(facecolor=color, alpha=0.15, 
                                            edgecolor=color, linewidth=1.2,
                                            label=f'{service.title()} service area'))
    
    # plt.suptitle(f'Temporal Evolution of Service Areas: {settl_name}', 
    #             fontsize=18, fontweight='bold', y=0.98)
    
    # Calculate optimal number of columns for legend
    n_legend_items = len(legend_elements)
    ncols = min(4, max(2, n_legend_items // 2))  # 2-4 columns depending on items
    
    # Add publication-quality legend at bottom
    legend = fig.legend(handles=legend_elements, 
                       loc='lower center', 
                       bbox_to_anchor=(0.5, -0.02),
                       fontsize=11,
                       frameon=True,
                       fancybox=False,
                       shadow=False,
                       framealpha=0.9,
                       edgecolor='black',
                       facecolor='white',
                       ncol=ncols,
                       columnspacing=1.5,
                       handletextpad=0.6,
                       handlelength=1.8,
                       borderpad=0.8)
    
    # Style the legend frame for publication quality
    legend.get_frame().set_linewidth(0.8)
    
    # Adjust layout to accommodate bottom legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.93)  # Make room for bottom legend and title
    
    return fig


def plot_stable_communities(all_results,  settl_name, 
                          month_range, figsize=(20, 8)):
    """
    Visualize the most stable communities across services and time
    """
    # Use global defaults if not provided

    # Identify stable communities
    multi_service, temporal_stable, super_stable = identify_stable_communities(
        all_results, settl_name, month_range
    )
    
    # Get positions
    pos = {}
    for service in service_list:
        try:
            for month_idx in month_range:
                graph = all_results[settl_name][service]["stats"].graphs[month_idx]
                for node, data in graph.nodes(data=True):
                    if node not in pos:
                        if "x" in data and "y" in data:
                            pos[node] = (data["x"], data["y"])
                        elif "longitude" in data and "latitude" in data:
                            pos[node] = (data["longitude"], data["latitude"])
                if pos:
                    break
        except:
            continue
        if pos:
            break
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Multi-service stable communities
    ax1 = axes[0]
    if pos:
        all_x = [pos[node][0] for node in pos]
        all_y = [pos[node][1] for node in pos]
        ax1.scatter(all_x, all_y, c='lightgray', s=10, alpha=0.3, zorder=1)
    
    if multi_service:
        # Color by service diversity
        for provider, comm_data in multi_service.items():
            if provider in pos:
                nodes = comm_data['nodes']
                node_pos = [pos[node] for node in nodes if node in pos]
                
                if len(node_pos) >= 3:
                    try:
                        points = np.array(node_pos)
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        
                        # Color intensity based on service diversity
                        color_intensity = plt.cm.YlOrRd(comm_data['service_diversity'])
                        
                        polygon = Polygon(hull_points, alpha=0.4,
                                        facecolor=color_intensity, 
                                        edgecolor='darkred', linewidth=2)
                        ax1.add_patch(polygon)
                        
                        # Label with number of services
                        centroid = np.mean(hull_points, axis=0)
                        ax1.text(centroid[0], centroid[1], 
                                f"{len(comm_data['services'])}",
                                fontsize=10, fontweight='bold',
                                ha='center', va='center',
                                bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
                    except:
                        pass
                
                # Mark provider
                ax1.scatter(pos[provider][0], pos[provider][1], 
                          c='red', s=100, marker='*',
                          edgecolors='black', linewidth=1, zorder=10)
    
    ax1.set_title(f'Service-Stable Communities\n(colored by service diversity)', fontsize=12)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Panel 2: Temporally stable communities
    ax2 = axes[1]
    if pos:
        ax2.scatter(all_x, all_y, c='lightgray', s=10, alpha=0.3, zorder=1)
    
    if temporal_stable:
        for provider, comm_data in temporal_stable.items():
            if provider in pos:
                nodes = comm_data['stable_nodes']
                node_pos = [pos[node] for node in nodes if node in pos]
                
                if len(node_pos) >= 3:
                    try:
                        points = np.array(node_pos)
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        
                        # Color intensity based on temporal stability
                        color_intensity = plt.cm.BuGn(comm_data['temporal_stability'])
                        
                        polygon = Polygon(hull_points, alpha=0.4,
                                        facecolor=color_intensity, 
                                        edgecolor='darkgreen', linewidth=2)
                        ax2.add_patch(polygon)
                        
                        # Label with stability score
                        centroid = np.mean(hull_points, axis=0)
                        ax2.text(centroid[0], centroid[1], 
                                f"{comm_data['temporal_stability']:.2f}",
                                fontsize=10, fontweight='bold',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
                
                # Mark provider
                ax2.scatter(pos[provider][0], pos[provider][1], 
                          c='green', s=100, marker='*',
                          edgecolors='black', linewidth=1, zorder=10)
    
    ax2.set_title(f'Temporally-Stable Communities\n(colored by temporal persistence)', fontsize=12)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Panel 3: Super-stable communities (stable across both dimensions)
    ax3 = axes[2]
    if pos:
        ax3.scatter(all_x, all_y, c='lightgray', s=10, alpha=0.3, zorder=1)
    
    if super_stable:
        # Sort by stability score
        sorted_communities = sorted(super_stable.items(), 
                                  key=lambda x: x[1]['stability_score'], 
                                  reverse=True)
        
        for rank, (provider, comm_data) in enumerate(sorted_communities[:5]):  # Top 5
            if provider in pos:
                nodes = comm_data['stable_nodes']
                node_pos = [pos[node] for node in nodes if node in pos]
                
                if len(node_pos) >= 3:
                    try:
                        points = np.array(node_pos)
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        
                        # Color by rank
                        colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#4B0082', '#8B008B']  # Gold, Silver, Bronze, etc.
                        color = colors[min(rank, 4)]
                        
                        polygon = Polygon(hull_points, alpha=0.5,
                                        facecolor=color, 
                                        edgecolor='black', linewidth=2.5)
                        ax3.add_patch(polygon)
                        
                        # Label with rank and score
                        centroid = np.mean(hull_points, axis=0)
                        ax3.text(centroid[0], centroid[1], 
                                f"#{rank+1}\n{comm_data['stability_score']:.2f}",
                                fontsize=10, fontweight='bold',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                    except:
                        pass
                
                # Mark provider with rank
                ax3.scatter(pos[provider][0], pos[provider][1], 
                          c='black', s=120, marker='*',
                          edgecolors='gold', linewidth=2, zorder=10)
    
    ax3.set_title(f'GenrallyStable Communities\n(Top 5 by combined stability)', fontsize=12)
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    plt.suptitle(f'Stable Community Analysis - {settl_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    return multi_service, temporal_stable, super_stable

