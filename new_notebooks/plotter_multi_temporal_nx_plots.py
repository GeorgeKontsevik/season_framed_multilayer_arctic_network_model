import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle
import matplotlib.patches as mpatches
from collections import defaultdict
from constants import SERVICE_COLORS, service_list, month_order

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

def plot_temporal_metrics(temporal_metrics, monthly_communities, figsize=(16, 10)):
    """
    Comprehensive visualization of temporal metrics
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Jaccard Similarity Timeline
    ax1 = axes[0, 0]
    if temporal_metrics['jaccard_similarity']:
        transitions = list(temporal_metrics['jaccard_similarity'].keys())
        values = list(temporal_metrics['jaccard_similarity'].values())
        ax1.plot(range(len(transitions)), values, 'o-', color='#3498db', linewidth=2, markersize=8)
        ax1.set_xticks(range(len(transitions)))
        ax1.set_xticklabels(transitions, rotation=45)
        ax1.set_ylabel('Jaccard Similarity')
        ax1.set_title('Community Similarity Between Consecutive Months')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
    
    # 2. NMI Scores
    ax2 = axes[0, 1]
    if temporal_metrics['nmi_scores']:
        transitions = list(temporal_metrics['nmi_scores'].keys())
        values = list(temporal_metrics['nmi_scores'].values())
        ax2.plot(range(len(transitions)), values, 's-', color='#e74c3c', linewidth=2, markersize=8)
        ax2.set_xticks(range(len(transitions)))
        ax2.set_xticklabels(transitions, rotation=45)
        ax2.set_ylabel('NMI Score')
        ax2.set_title('Normalized Mutual Information')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    
    # 3. Persistence Coefficient
    ax3 = axes[0, 2]
    if temporal_metrics['persistence_coefficient']:
        transitions = list(temporal_metrics['persistence_coefficient'].keys())
        values = list(temporal_metrics['persistence_coefficient'].values())
        ax3.plot(range(len(transitions)), values, '^-', color='#2ecc71', linewidth=2, markersize=8)
        ax3.set_xticks(range(len(transitions)))
        ax3.set_xticklabels(transitions, rotation=45)
        ax3.set_ylabel('Persistence Coefficient')
        ax3.set_title('Community Persistence')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
    
    # 4. Autarky Evolution
    ax4 = axes[1, 0]
    if temporal_metrics['autarky_evolution']:
        months = sorted(temporal_metrics['autarky_evolution'].keys())
        values = [temporal_metrics['autarky_evolution'][m] for m in months]
        ax4.bar(months, values, color='#9b59b6', alpha=0.7)
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Autarky Coefficient')
        ax4.set_title('Network Autarky Evolution')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Modularity Evolution
    ax5 = axes[1, 1]
    if temporal_metrics['modularity_evolution']:
        months = sorted(temporal_metrics['modularity_evolution'].keys())
        values = [temporal_metrics['modularity_evolution'][m] for m in months]
        ax5.plot(months, values, 'o-', color='#f39c12', linewidth=2, markersize=8)
        ax5.fill_between(months, values, alpha=0.3, color='#f39c12')
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Modularity Q')
        ax5.set_title('Network Modularity Evolution')
        ax5.grid(True, alpha=0.3)
    
    # 6. Community Evolution Stacked Bar
    ax6 = axes[1, 2]
    if temporal_metrics['community_evolution']:
        evolution_data = defaultdict(list)
        transitions = sorted(temporal_metrics['community_evolution'].keys())
        
        for transition in transitions:
            for event_type in ['stable', 'grown', 'shrunk', 'split', 'merged', 'disappeared', 'new']:
                evolution_data[event_type].append(
                    temporal_metrics['community_evolution'][transition].get(event_type, 0)
                )
        
        bottom = np.zeros(len(transitions))
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#95a5a6', '#1abc9c']
        
        for (event_type, values), color in zip(evolution_data.items(), colors):
            ax6.bar(range(len(transitions)), values, bottom=bottom, 
                   label=event_type.capitalize(), color=color, alpha=0.8)
            bottom += np.array(values)
        
        ax6.set_xticks(range(len(transitions)))
        ax6.set_xticklabels(transitions, rotation=45)
        ax6.set_ylabel('Number of Communities')
        ax6.set_title('Community Evolution Events')
        ax6.legend(loc='upper right', fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Temporal Network Analysis Metrics', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    return fig

def create_temporal_summary_report(temporal_metrics, monthly_communities):
    """
    Generate a text summary of temporal analysis
    """
    report = []
    report.append("=" * 60)
    report.append("TEMPORAL MULTILAYER NETWORK ANALYSIS REPORT")
    report.append("=" * 60)
    
    # Average metrics
    if temporal_metrics['jaccard_similarity']:
        avg_jaccard = np.mean(list(temporal_metrics['jaccard_similarity'].values()))
        report.append(f"\n📊 TEMPORAL STABILITY METRICS:")
        report.append(f"   Average Jaccard Similarity: {avg_jaccard:.3f}")
        report.append(f"   Interpretation: {'High' if avg_jaccard > 0.7 else 'Moderate' if avg_jaccard > 0.4 else 'Low'} temporal stability")
    
    if temporal_metrics['nmi_scores']:
        avg_nmi = np.mean(list(temporal_metrics['nmi_scores'].values()))
        report.append(f"   Average NMI Score: {avg_nmi:.3f}")
    
    if temporal_metrics['persistence_coefficient']:
        avg_persistence = np.mean(list(temporal_metrics['persistence_coefficient'].values()))
        report.append(f"   Average Persistence: {avg_persistence:.3f}")
    
    # Autarky trend
    if temporal_metrics['autarky_evolution']:
        autarky_values = list(temporal_metrics['autarky_evolution'].values())
        trend = "increasing" if autarky_values[-1] > autarky_values[0] else "decreasing"
        report.append(f"\n🔄 SELF-SUFFICIENCY EVOLUTION:")
        report.append(f"   Autarky trend: {trend}")
        report.append(f"   Initial: {autarky_values[0]:.3f}, Final: {autarky_values[-1]:.3f}")
    
    # Community dynamics
    if temporal_metrics['community_evolution']:
        total_events = defaultdict(int)
        for evolution in temporal_metrics['community_evolution'].values():
            for event_type, count in evolution.items():
                total_events[event_type] += count
        
        report.append(f"\n🌐 COMMUNITY DYNAMICS:")
        report.append(f"   Stable communities: {total_events['stable']}")
        report.append(f"   Growing communities: {total_events['grown']}")
        report.append(f"   Shrinking communities: {total_events['shrunk']}")
        report.append(f"   Split events: {total_events['split']}")
        report.append(f"   Merge events: {total_events['merged']}")
        report.append(f"   New communities: {total_events['new']}")
        report.append(f"   Disappeared: {total_events['disappeared']}")
    
    # Key findings
    report.append(f"\n🔍 KEY FINDINGS:")
    if avg_jaccard > 0.7:
        report.append("   ✓ Strong temporal persistence of service communities")
    elif avg_jaccard > 0.4:
        report.append("   ⚠ Moderate temporal variation in community structure")
    else:
        report.append("   ⚠ High temporal volatility in service provision")
    
    if trend == "increasing":
        report.append("   ✓ Improving self-sufficiency over time")
    else:
        report.append("   ⚠ Declining self-sufficiency requires attention")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


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
    print("\n🏛️ Stable communities...")
    multi, temporal, super_stable = plot_stable_communities(all_results, settl_name, month_range)
    plt.show()
    
    print(f"   Multi-service: {len(multi)} | Temporal: {len(temporal)} | Super: {len(super_stable)}")
    
    if super_stable:
        top3 = sorted(super_stable.items(), key=lambda x: x[1]['stability_score'], reverse=True)[:3]
        for i, (p, d) in enumerate(top3, 1):
            print(f"   #{i} {p}: {len(d['services'])} services, score={d['stability_score']:.2f}")
    
    # 3. Metrics
    print("\n📊 Temporal metrics...")
    metrics, communities = calculate_temporal_metrics(all_results, settl_name, month_range)
    plot_temporal_metrics(metrics, communities)
    plt.show()
    
    # 4. Report
    print(create_temporal_summary_report(metrics, communities))
    
    return {
        'metrics': metrics,
        'communities': communities,
        'multi_service': multi,
        'temporal_stable': temporal,
        'super_stable': super_stable
    }

def quick_single_month_analysis(all_results, settl_name, month_idx=5):
    """Quick single month visualization"""
    print(f"\n📍 Month: {month_order[month_idx]}")
    
    pmap, provs = plot_enhanced_service_areas(all_results, settl_name, month_idx=month_idx)
    plt.show()
    
    print(f"   Providers: {sum(len(p) for p in provs.values())} | Services: {len(provs)}")
    return pmap, provs

def quick_stability_check(all_results, settl_name, month_range):
    """Quick stability analysis only"""
    multi, temporal, super_stable = identify_stable_communities(all_results, settl_name, month_range)
    
    print(f"\n🎯 Stability Summary:")
    print(f"   Multi-service providers: {len(multi)}")
    print(f"   Temporally stable: {len(temporal)}")
    print(f"   Super-stable: {len(super_stable)}")
    
    if super_stable:
        best = max(super_stable.items(), key=lambda x: x[1]['stability_score'])
        print(f"   Best: {best[0]} (score={best[1]['stability_score']:.2f})")
    
    return multi, temporal, super_stable

def compare_months(all_results, settl_name, month1=5, month2=8):
    """Compare two specific months"""
    from collections import defaultdict
    
    comms1, comms2 = defaultdict(set), defaultdict(set)
    
    for service in service_list:
        try:
            for m, comm in [(month1, comms1), (month2, comms2)]:
                g = all_results[settl_name][service]["stats"].graphs[m]
                for s, t, d in g.edges(data=True):
                    if d.get("is_service_flow") and d.get("assignment", 0) > 0 and s != t:
                        comm[t].update([s, t])
        except:
            continue
    
    all1, all2 = set().union(*comms1.values()), set().union(*comms2.values())
    jaccard = len(all1 & all2) / len(all1 | all2) if all1 | all2 else 0
    
    print(f"\n🔄 {month_order[month1]} vs {month_order[month2]}:")
    print(f"   Providers: {len(comms1)} → {len(comms2)}")
    print(f"   Coverage: {len(all1)} → {len(all2)} nodes")
    print(f"   Similarity: {jaccard:.2%}")
    
    return jaccard, comms1, comms2


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

def plot_temporal_service_evolution(all_results,  settl_name, 
                                   month_range, figsize=(20, 12)):
    """
    Visualize service areas evolution across all months in the range
    """
    # Use global defaults if not provided

    
    months = list(month_range)
    n_months = len(months)
    
    # Create subplots grid
    cols = 3
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
    
    # Plot for each month
    for plot_idx, month_idx in enumerate(months):
        ax = axes[plot_idx]
        
        # Base nodes
        if all_positions:
            all_x = [all_positions[node][0] for node in all_positions]
            all_y = [all_positions[node][1] for node in all_positions]
            ax.scatter(all_x, all_y, c='lightgray', s=10, alpha=0.3, zorder=1)
        
        provider_count = 0
        service_coverage = set()
        
        # Plot each service
        for service in service_list:
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
                for provider, consumers in providers.items():
                    if len(consumers) >= 1:
                        group_nodes = consumers | {provider}
                        group_pos = [all_positions[node] for node in group_nodes if node in all_positions]
                        
                        if len(group_pos) >= 3:
                            try:
                                points = np.array(group_pos)
                                hull = ConvexHull(points)
                                hull_points = points[hull.vertices]
                                
                                polygon = Polygon(hull_points, alpha=0.1, 
                                                facecolor=color, edgecolor=color,
                                                linewidth=1, zorder=2)
                                ax.add_patch(polygon)
                            except:
                                pass
                        
                        # Mark provider
                        if provider in all_positions:
                            ax.scatter(all_positions[provider][0], all_positions[provider][1], 
                                     c=color, s=50, marker='*', 
                                     edgecolors='white', linewidth=0.5, zorder=10)
                            provider_count += 1
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
    
    plt.suptitle(f'Temporal Evolution of Service Areas - {settl_name}', fontsize=16)
    plt.tight_layout()
    return fig

def identify_stable_communities(all_results, settl_name, 
                               month_range, stability_threshold=0.5):
    """
    Identify communities that are stable across services and time
    """

    
    # Track community membership across services and time
    service_communities = defaultdict(lambda: defaultdict(set))  # service -> provider -> nodes
    temporal_communities = defaultdict(lambda: defaultdict(set))  # month -> provider -> nodes
    
    for month_idx in month_range:
        monthly_providers = defaultdict(set)
        
        for service in service_list:
            try:
                graph = all_results[settl_name][service]["stats"].graphs[month_idx]
                
                # Extract provider-consumer relationships
                for source, target, data in graph.edges(data=True):
                    if (data.get("is_service_flow", False) and 
                        data.get("assignment", 0) > 0 and 
                        source != target):
                        # Service-level tracking
                        service_communities[service][target].add(source)
                        service_communities[service][target].add(target)
                        
                        # Temporal tracking
                        temporal_communities[month_idx][target].add(source)
                        temporal_communities[month_idx][target].add(target)
                        monthly_providers[target].add(service)
                        
            except Exception as e:
                print(f"Error processing service {service} for month {month_idx}: {e}")
                # continue
    
    # Find stable communities across services (multi-service providers)
    multi_service_communities = {}
    for provider in set().union(*[set(sc.keys()) for sc in service_communities.values()]):
        services_provided = []
        community_nodes = set()
        
        for service in service_list:
            if provider in service_communities[service]:
                services_provided.append(service)
                community_nodes.update(service_communities[service][provider])
        
        if len(services_provided) >= 2:  # Provider serves multiple services
            multi_service_communities[provider] = {
                'services': services_provided,
                'nodes': community_nodes,
                'service_diversity': len(services_provided) / len(service_list)
            }
    
    # Find temporally stable communities
    temporally_stable_communities = {}
    all_providers = set().union(*[set(tc.keys()) for tc in temporal_communities.values()])
    
    for provider in all_providers:
        presence_months = []
        stable_nodes = None
        
        for month_idx in month_range:
            if provider in temporal_communities[month_idx]:
                presence_months.append(month_idx)
                if stable_nodes is None:
                    stable_nodes = temporal_communities[month_idx][provider].copy()
                else:
                    stable_nodes &= temporal_communities[month_idx][provider]
        
        temporal_stability = len(presence_months) / len(list(month_range))
        
        if temporal_stability >= stability_threshold and stable_nodes:
            temporally_stable_communities[provider] = {
                'months_present': presence_months,
                'stable_nodes': stable_nodes,
                'temporal_stability': temporal_stability,
                'node_retention': len(stable_nodes) / max([len(temporal_communities[m][provider]) 
                                                           for m in presence_months])
            }
    
    # Find communities stable across both dimensions
    super_stable_communities = {}
    for provider in set(multi_service_communities.keys()) & set(temporally_stable_communities.keys()):
        super_stable_communities[provider] = {
            'services': multi_service_communities[provider]['services'],
            'stable_nodes': temporally_stable_communities[provider]['stable_nodes'],
            'service_diversity': multi_service_communities[provider]['service_diversity'],
            'temporal_stability': temporally_stable_communities[provider]['temporal_stability'],
            'stability_score': (multi_service_communities[provider]['service_diversity'] + 
                              temporally_stable_communities[provider]['temporal_stability']) / 2
        }


    return multi_service_communities, temporally_stable_communities, super_stable_communities

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
    
    ax3.set_title(f'Super-Stable Communities\n(Top 5 by combined stability)', fontsize=12)
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    plt.suptitle(f'Stable Community Analysis - {settl_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    return multi_service, temporal_stable, super_stable

def calculate_temporal_metrics(all_results, settl_name, month_range):
    """
    Calculate comprehensive temporal metrics based on literature review
    """
    # Use global defaults if not provided

    temporal_metrics = {
        'jaccard_similarity': {},
        'nmi_scores': {},
        'persistence_coefficient': {},
        'autarky_evolution': {},
        'modularity_evolution': {},
        'community_evolution': {}
    }
    
    # Store communities for each month
    monthly_communities = {}
    monthly_graphs = {}
    
    for month_idx in month_range:
        # Extract provider-consumer communities
        communities = defaultdict(set)
        combined_graph = nx.Graph()
        
        for service in service_list:
            try:
                graph = all_results[settl_name][service]["stats"].graphs[month_idx]
                
                # Add to combined graph for modularity calculation
                for u, v, data in graph.edges(data=True):
                    if data.get("is_service_flow", False) and data.get("assignment", 0) > 0:
                        combined_graph.add_edge(u, v, weight=data.get("assignment", 1),
                                              service=service)
                
                # Extract communities
                for source, target, data in graph.edges(data=True):
                    if (data.get("is_service_flow", False) and 
                        data.get("assignment", 0) > 0 and 
                        source != target):
                        communities[target].add(source)
                        communities[target].add(target)
                        
            except:
                continue
        
        monthly_communities[month_idx] = communities
        monthly_graphs[month_idx] = combined_graph
    
    # Calculate metrics between consecutive months
    months = sorted(monthly_communities.keys())
    
    for i in range(len(months) - 1):
        month1, month2 = months[i], months[i + 1]
        
        # 1. Jaccard Similarity
        comm1 = set().union(*monthly_communities[month1].values()) if monthly_communities[month1] else set()
        comm2 = set().union(*monthly_communities[month2].values()) if monthly_communities[month2] else set()
        
        if comm1 and comm2:
            jaccard = len(comm1 & comm2) / len(comm1 | comm2)
            temporal_metrics['jaccard_similarity'][f'{month_order[month1]}->{month_order[month2]}'] = jaccard
        
        # 2. Persistence Coefficient
        persistence_scores = []
        for provider, community in monthly_communities[month1].items():
            if provider in monthly_communities[month2]:
                next_community = monthly_communities[month2][provider]
                if community and next_community:
                    persistence = len(community & next_community) / len(community)
                    persistence_scores.append(persistence)
        
        if persistence_scores:
            key = f'{month_order[month1]}->{month_order[month2]}'
            temporal_metrics['persistence_coefficient'][key] = np.mean(persistence_scores)
        
        # 3. NMI Score
        nmi = calculate_nmi(monthly_communities[month1], monthly_communities[month2])
        temporal_metrics['nmi_scores'][f'{month_order[month1]}->{month_order[month2]}'] = nmi
        
        # 4. Community Evolution
        evolution = analyze_community_evolution(monthly_communities[month1], monthly_communities[month2])
        temporal_metrics['community_evolution'][f'{month_order[month1]}->{month_order[month2]}'] = evolution
    
    # Calculate per-month metrics
    for month_idx in months:
        # Autarky coefficient
        if monthly_graphs[month_idx].number_of_edges() > 0:
            internal_flows = 0
            total_flows = 0
            
            for provider, community in monthly_communities[month_idx].items():
                for node in community:
                    for neighbor in monthly_graphs[month_idx].neighbors(node):
                        total_flows += 1
                        if neighbor in community:
                            internal_flows += 1
            
            autarky = internal_flows / (2 * total_flows) if total_flows > 0 else 0
            temporal_metrics['autarky_evolution'][month_order[month_idx]] = autarky
        
        # Modularity
        if monthly_graphs[month_idx].number_of_nodes() > 0:
            partition = {}
            for comm_id, (provider, nodes) in enumerate(monthly_communities[month_idx].items()):
                for node in nodes:
                    partition[node] = comm_id
            
            if partition:
                Q = calculate_modularity(monthly_graphs[month_idx], partition)
                temporal_metrics['modularity_evolution'][month_order[month_idx]] = Q
    
    return temporal_metrics, monthly_communities

def calculate_nmi(partition1, partition2):
    """
    Calculate Normalized Mutual Information between two partitions
    """
    # Get all nodes
    all_nodes = set()
    for nodes in partition1.values():
        all_nodes.update(nodes)
    for nodes in partition2.values():
        all_nodes.update(nodes)
    
    if not all_nodes:
        return 0
    
    # Create cluster assignments
    labels1 = np.zeros(len(all_nodes))
    labels2 = np.zeros(len(all_nodes))
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    
    for comm_id, (provider, nodes) in enumerate(partition1.items()):
        for node in nodes:
            if node in node_to_idx:
                labels1[node_to_idx[node]] = comm_id
    
    for comm_id, (provider, nodes) in enumerate(partition2.items()):
        for node in nodes:
            if node in node_to_idx:
                labels2[node_to_idx[node]] = comm_id
    
    # Calculate mutual information
    h1 = entropy(np.bincount(labels1.astype(int)))
    h2 = entropy(np.bincount(labels2.astype(int)))
    
    if h1 == 0 or h2 == 0:
        return 0
    
    # Joint histogram
    joint_hist = np.histogram2d(labels1, labels2, 
                                bins=[len(np.unique(labels1)), len(np.unique(labels2))])[0]
    mi = 0
    for i in range(joint_hist.shape[0]):
        for j in range(joint_hist.shape[1]):
            if joint_hist[i, j] > 0:
                mi += joint_hist[i, j] * np.log(joint_hist[i, j] * len(all_nodes) / 
                                               (np.sum(joint_hist[i, :]) * np.sum(joint_hist[:, j])))
    
    mi /= len(all_nodes)
    nmi = 2 * mi / (h1 + h2)
    
    return nmi

def analyze_community_evolution(communities_t1, communities_t2):
    """
    Analyze how communities evolve: growth, split, merge, stable
    """
    evolution = {
        'stable': 0,
        'grown': 0,
        'shrunk': 0,
        'split': 0,
        'merged': 0,
        'disappeared': 0,
        'new': 0
    }
    
    # Match communities between time steps
    for provider1, comm1 in communities_t1.items():
        best_match = None
        best_overlap = 0
        
        for provider2, comm2 in communities_t2.items():
            overlap = len(comm1 & comm2) / len(comm1 | comm2) if len(comm1 | comm2) > 0 else 0
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = provider2
        
        if best_match:
            if best_overlap > 0.8:
                if len(communities_t2[best_match]) > len(comm1) * 1.1:
                    evolution['grown'] += 1
                elif len(communities_t2[best_match]) < len(comm1) * 0.9:
                    evolution['shrunk'] += 1
                else:
                    evolution['stable'] += 1
            elif best_overlap > 0.3:
                evolution['split'] += 1
        else:
            evolution['disappeared'] += 1
    
    # Check for new communities
    matched_providers = set()
    for provider1, comm1 in communities_t1.items():
        for provider2, comm2 in communities_t2.items():
            overlap = len(comm1 & comm2) / len(comm1 | comm2) if len(comm1 | comm2) > 0 else 0
            if overlap > 0.3:
                matched_providers.add(provider2)
    
    for provider2 in communities_t2:
        if provider2 not in matched_providers:
            evolution['new'] += 1
    
    return evolution

def calculate_modularity(G, partition):
    """
    Calculate modularity Q for a given partition
    """
    if G.number_of_edges() == 0:
        return 0
    
    m = G.number_of_edges()
    Q = 0
    
    for node1 in G.nodes():
        for node2 in G.nodes():
            if partition.get(node1) == partition.get(node2):
                # Same community
                A_ij = 1 if G.has_edge(node1, node2) else 0
                k_i = G.degree(node1)
                k_j = G.degree(node2)
                Q += A_ij - (k_i * k_j) / (2 * m)
    
    return Q / (2 * m)

def plot_publication_service_areas(all_results,  settl_name, 
                                  month_idx=5, figsize=(10, 8), dpi=300):
    """
    Publication-ready visualization with consistent color scheme
    """
    
    # Get positions and relationships
    pos = {}
    provider_consumer_map = defaultdict(lambda: defaultdict(set))
    
    for service in service_list:
        try:
            graph = all_results[settl_name][service]["stats"].graphs[month_idx]
            if not pos:
                for node, data in graph.nodes(data=True):
                    if "x" in data and "y" in data:
                        pos[node] = (data["x"], data["y"])
                    elif "longitude" in data and "latitude" in data:
                        pos[node] = (data["longitude"], data["latitude"])
            
            for source, target, data in graph.edges(data=True):
                if (data.get("is_service_flow", False) and 
                    data.get("assignment", 0) > 0 and source != target):
                    provider_consumer_map[service][target].add(source)
        except:
            continue
    
    if not pos:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Background nodes
    all_x = [pos[node][0] for node in pos]
    all_y = [pos[node][1] for node in pos]
    ax.scatter(all_x, all_y, c='#e0e0e0', s=8, alpha=0.5, zorder=1, rasterized=True)
    
    # Plot service areas with publication colors
    legend_handles = []
    
    for service_idx, service in enumerate(service_list):
        if service not in provider_consumer_map:
            continue
            
        color = SERVICE_COLORS.get(service, "#888888")
        service_plotted = False
        
        for provider, consumers in provider_consumer_map[service].items():
            if len(consumers) >= 1:
                group_nodes = consumers | {provider}
                group_pos = [pos[node] for node in group_nodes if node in pos]
                
                # Draw convex hull
                if len(group_pos) >= 3:
                    try:
                        from scipy.spatial import ConvexHull
                        points = np.array(group_pos)
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        
                        polygon = Polygon(hull_points, alpha=0.2, 
                                        facecolor=color, edgecolor=color,
                                        linewidth=1.0, zorder=2 + service_idx * 0.1)
                        ax.add_patch(polygon)
                    except:
                        pass
                
                # Plot nodes
                consumer_x = [pos[node][0] for node in consumers if node in pos]
                consumer_y = [pos[node][1] for node in consumers if node in pos]
                
                ax.scatter(consumer_x, consumer_y, c=color, s=20, alpha=0.7,
                          edgecolors='white', linewidth=0.3, zorder=10, rasterized=True)
                
                # Provider marker
                if provider in pos:
                    ax.scatter(pos[provider][0], pos[provider][1], 
                              c=color, s=80, marker='*',
                              edgecolors='black', linewidth=0.8, zorder=12)
                    service_plotted = True
        
        # Add to legend
        if service_plotted:
            legend_handles.append(mpatches.Patch(color=color, label=service.capitalize(), alpha=0.7))
    
    # Title and formatting
    month_name = month_order[month_idx] if month_idx < len(month_order) else f"Month {month_idx}"
    ax.set_title(f'Service Provision Areas - {settl_name.replace("_", " ").title()} ({month_name})', 
                fontweight='bold', pad=15)
    
    # Legend
    ax.legend(handles=legend_handles, loc='upper left', frameon=True, 
             fancybox=False, shadow=False, framealpha=0.9, ncol=2)
    
    # Clean axes
    ax.set_xlabel('Longitude', fontweight='bold')
    ax.set_ylabel('Latitude', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    
    # Tighten layout
    plt.tight_layout()
    
    return fig, ax

def plot_publication_temporal_metrics(temporal_metrics, figsize=(12, 8), dpi=300):
    """
    Publication-ready temporal metrics visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    
    # Define consistent colors for metrics
    metric_colors = {
        'jaccard': '#2E86AB',      # Deep blue
        'nmi': '#A23B72',           # Purple-red
        'persistence': '#2A9D8F',   # Teal
        'autarky': '#6A4C93',       # Purple
        'modularity': '#F77F00',    # Orange
        'evolution': SERVICE_COLORS  # Use service colors
    }
    
    # 1. Jaccard Similarity
    ax1 = axes[0, 0]
    if temporal_metrics['jaccard_similarity']:
        transitions = list(temporal_metrics['jaccard_similarity'].keys())
        values = list(temporal_metrics['jaccard_similarity'].values())
        ax1.plot(range(len(transitions)), values, 'o-', color=metric_colors['jaccard'], 
                linewidth=2.5, markersize=7, markeredgecolor='white', markeredgewidth=1)
        ax1.set_xticks(range(len(transitions)))
        ax1.set_xticklabels(transitions, rotation=45, ha='right')
        ax1.set_ylabel('Jaccard Index', fontweight='bold')
        ax1.set_title('(a) Community Similarity', fontweight='bold')
        ax1.set_ylim([0, 1.05])
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # 2. NMI
    ax2 = axes[0, 1]
    if temporal_metrics['nmi_scores']:
        transitions = list(temporal_metrics['nmi_scores'].keys())
        values = list(temporal_metrics['nmi_scores'].values())
        ax2.plot(range(len(transitions)), values, 's-', color=metric_colors['nmi'], 
                linewidth=2.5, markersize=7, markeredgecolor='white', markeredgewidth=1)
        ax2.set_xticks(range(len(transitions)))
        ax2.set_xticklabels(transitions, rotation=45, ha='right')
        ax2.set_ylabel('NMI Score', fontweight='bold')
        ax2.set_title('(b) Mutual Information', fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # 3. Persistence
    ax3 = axes[0, 2]
    if temporal_metrics['persistence_coefficient']:
        transitions = list(temporal_metrics['persistence_coefficient'].keys())
        values = list(temporal_metrics['persistence_coefficient'].values())
        ax3.plot(range(len(transitions)), values, '^-', color=metric_colors['persistence'], 
                linewidth=2.5, markersize=7, markeredgecolor='white', markeredgewidth=1)
        ax3.set_xticks(range(len(transitions)))
        ax3.set_xticklabels(transitions, rotation=45, ha='right')
        ax3.set_ylabel('Persistence', fontweight='bold')
        ax3.set_title('(c) Community Persistence', fontweight='bold')
        ax3.set_ylim([0, 1.05])
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # 4. Autarky
    ax4 = axes[1, 0]
    if temporal_metrics['autarky_evolution']:
        months = list(temporal_metrics['autarky_evolution'].keys())
        values = list(temporal_metrics['autarky_evolution'].values())
        bars = ax4.bar(range(len(months)), values, color=metric_colors['autarky'], 
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_xticks(range(len(months)))
        ax4.set_xticklabels(months, rotation=45, ha='right')
        ax4.set_ylabel('Autarky Coefficient', fontweight='bold')
        ax4.set_title('(d) Self-Sufficiency Evolution', fontweight='bold')
        ax4.set_ylim([0, max(values) * 1.1 if values else 0.5])
    
    # 5. Modularity
    ax5 = axes[1, 1]
    if temporal_metrics['modularity_evolution']:
        months = list(temporal_metrics['modularity_evolution'].keys())
        values = list(temporal_metrics['modularity_evolution'].values())
        ax5.plot(range(len(months)), values, 'o-', color=metric_colors['modularity'], 
                linewidth=2.5, markersize=7, markeredgecolor='white', markeredgewidth=1)
        ax5.fill_between(range(len(months)), values, alpha=0.3, color=metric_colors['modularity'])
        ax5.set_xticks(range(len(months)))
        ax5.set_xticklabels(months, rotation=45, ha='right')
        ax5.set_ylabel('Modularity Q', fontweight='bold')
        ax5.set_title('(e) Network Modularity', fontweight='bold')
        ax5.set_ylim([0, max(values) * 1.1 if values else 0.5])
    
    # 6. Evolution Events
    ax6 = axes[1, 2]
    if temporal_metrics['community_evolution']:
        from collections import defaultdict
        evolution_data = defaultdict(list)
        transitions = list(temporal_metrics['community_evolution'].keys())
        
        event_types = ['stable', 'grown', 'shrunk', 'split', 'merged', 'disappeared', 'new']
        event_colors = ['#2A9D8F', '#52B788', '#95D5B2', '#F77F00', '#FCBF49', '#D62828', '#003049']
        
        for transition in transitions:
            for event_type in event_types:
                evolution_data[event_type].append(
                    temporal_metrics['community_evolution'][transition].get(event_type, 0)
                )
        
        bottom = np.zeros(len(transitions))
        for (event_type, values), color in zip(evolution_data.items(), event_colors):
            ax6.bar(range(len(transitions)), values, bottom=bottom, 
                   label=event_type.capitalize(), color=color, alpha=0.85,
                   edgecolor='black', linewidth=0.5)
            bottom += np.array(values)
        
        ax6.set_xticks(range(len(transitions)))
        ax6.set_xticklabels(transitions, rotation=45, ha='right')
        ax6.set_ylabel('Number of Events', fontweight='bold')
        ax6.set_title('(f) Community Dynamics', fontweight='bold')
        ax6.legend(loc='upper right', fontsize=8, ncol=2, frameon=True, framealpha=0.9)
    
    # Overall title
    fig.suptitle('Temporal Network Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def save_publication_figures(all_results, output_dir='./figures/', format='pdf'):
    """
    Generate and save all publication-ready figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Service areas for key month
    fig1, _ = plot_publication_service_areas(all_results, month_idx=5, dpi=300)
    if fig1:
        fig1.savefig(f'{output_dir}service_areas.{format}', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: service_areas.{format}")
    
    # 2. Temporal metrics
    metrics, _ = calculate_temporal_metrics(all_results)
    fig2 = plot_publication_temporal_metrics(metrics, dpi=300)
    if fig2:
        fig2.savefig(f'{output_dir}temporal_metrics.{format}', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: temporal_metrics.{format}")
    
    plt.show()
    print(f"\n📊 All figures saved to {output_dir}")

# Quick usage for publication
def quick_publication_plots(all_results):
    """Generate publication-ready plots with one command"""
    print("🎨 Generating publication-ready visualizations...")
    
    # Single month view
    fig1, ax1 = plot_publication_service_areas(all_results, month_idx=5)
    plt.show()
    
    # Temporal metrics
    metrics, _ = calculate_temporal_metrics(all_results)
    fig2 = plot_publication_temporal_metrics(metrics)
    plt.show()
    
    return fig1, fig2

# Example usage:





def plot_temporal_service_evolution(all_results, settl_name, 
                                   month_range, figsize=(20, 12)):
    """
    Visualize service areas evolution across all months in the range
    """

    
    months = list(month_range)
    n_months = len(months)
    
    # Create subplots grid
    cols = 3
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
    
    # Plot for each month
    for plot_idx, month_idx in enumerate(months):
        ax = axes[plot_idx]
        
        # Base nodes
        if all_positions:
            all_x = [all_positions[node][0] for node in all_positions]
            all_y = [all_positions[node][1] for node in all_positions]
            ax.scatter(all_x, all_y, c='lightgray', s=10, alpha=0.3, zorder=1)
        
        provider_count = 0
        service_coverage = set()
        
        # Plot each service
        for service in service_list:
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
                for provider, consumers in providers.items():
                    if len(consumers) >= 1:
                        group_nodes = consumers | {provider}
                        group_pos = [all_positions[node] for node in group_nodes if node in all_positions]
                        
                        if len(group_pos) >= 3:
                            try:
                                points = np.array(group_pos)
                                hull = ConvexHull(points)
                                hull_points = points[hull.vertices]
                                
                                polygon = Polygon(hull_points, alpha=0.1, 
                                                facecolor=color, edgecolor=color,
                                                linewidth=1, zorder=2)
                                ax.add_patch(polygon)
                            except:
                                pass
                        
                        # Mark provider
                        if provider in all_positions:
                            ax.scatter(all_positions[provider][0], all_positions[provider][1], 
                                     c=color, s=50, marker='*', 
                                     edgecolors='white', linewidth=0.5, zorder=10)
                            provider_count += 1
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
    
    plt.suptitle(f'Temporal Evolution of Service Areas - {settl_name}', fontsize=16)
    plt.tight_layout()
    return fig



def calculate_temporal_metrics(all_results, settl_name, month_range):
    """
    Calculate comprehensive temporal metrics based on literature review
    """
    # Use global defaults if not provide
    
    temporal_metrics = {
        'jaccard_similarity': {},
        'nmi_scores': {},
        'persistence_coefficient': {},
        'autarky_evolution': {},
        'modularity_evolution': {},
        'community_evolution': {}
    }
    
    # Store communities for each month
    monthly_communities = {}
    monthly_graphs = {}
    
    for month_idx in month_range:
        # Extract provider-consumer communities
        communities = defaultdict(set)
        combined_graph = nx.Graph()
        
        for service in service_list:
            try:
                graph = all_results[settl_name][service]["stats"].graphs[month_idx]
                
                # Add to combined graph for modularity calculation
                for u, v, data in graph.edges(data=True):
                    if data.get("is_service_flow", False) and data.get("assignment", 0) > 0:
                        combined_graph.add_edge(u, v, weight=data.get("assignment", 1),
                                              service=service)
                
                # Extract communities
                for source, target, data in graph.edges(data=True):
                    if (data.get("is_service_flow", False) and 
                        data.get("assignment", 0) > 0 and 
                        source != target):
                        communities[target].add(source)
                        communities[target].add(target)
                        
            except:
                continue
        
        monthly_communities[month_idx] = communities
        monthly_graphs[month_idx] = combined_graph
    
    # Calculate metrics between consecutive months
    months = sorted(monthly_communities.keys())
    
    for i in range(len(months) - 1):
        month1, month2 = months[i], months[i + 1]
        
        # 1. Jaccard Similarity
        comm1 = set().union(*monthly_communities[month1].values()) if monthly_communities[month1] else set()
        comm2 = set().union(*monthly_communities[month2].values()) if monthly_communities[month2] else set()
        
        if comm1 and comm2:
            jaccard = len(comm1 & comm2) / len(comm1 | comm2)
            temporal_metrics['jaccard_similarity'][f'{month_order[month1]}->{month_order[month2]}'] = jaccard
        
        # 2. Persistence Coefficient
        persistence_scores = []
        for provider, community in monthly_communities[month1].items():
            if provider in monthly_communities[month2]:
                next_community = monthly_communities[month2][provider]
                if community and next_community:
                    persistence = len(community & next_community) / len(community)
                    persistence_scores.append(persistence)
        
        if persistence_scores:
            key = f'{month_order[month1]}->{month_order[month2]}'
            temporal_metrics['persistence_coefficient'][key] = np.mean(persistence_scores)
        
        # 3. NMI Score
        nmi = calculate_nmi(monthly_communities[month1], monthly_communities[month2])
        temporal_metrics['nmi_scores'][f'{month_order[month1]}->{month_order[month2]}'] = nmi
        
        # 4. Community Evolution
        evolution = analyze_community_evolution(monthly_communities[month1], monthly_communities[month2])
        temporal_metrics['community_evolution'][f'{month_order[month1]}->{month_order[month2]}'] = evolution
    
    # Calculate per-month metrics
    for month_idx in months:
        # Autarky coefficient
        if monthly_graphs[month_idx].number_of_edges() > 0:
            internal_flows = 0
            total_flows = 0
            
            for provider, community in monthly_communities[month_idx].items():
                for node in community:
                    for neighbor in monthly_graphs[month_idx].neighbors(node):
                        total_flows += 1
                        if neighbor in community:
                            internal_flows += 1
            
            autarky = internal_flows / (2 * total_flows) if total_flows > 0 else 0
            temporal_metrics['autarky_evolution'][month_order[month_idx]] = autarky
        
        # Modularity
        if monthly_graphs[month_idx].number_of_nodes() > 0:
            partition = {}
            for comm_id, (provider, nodes) in enumerate(monthly_communities[month_idx].items()):
                for node in nodes:
                    partition[node] = comm_id
            
            if partition:
                Q = calculate_modularity(monthly_graphs[month_idx], partition)
                temporal_metrics['modularity_evolution'][month_order[month_idx]] = Q
    
    return temporal_metrics, monthly_communities
