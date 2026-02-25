import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle
import matplotlib.patches as mpatches
from collections import defaultdict
from scripts.preprocesser.constants import SERVICE_COLORS

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from collections import defaultdict
import seaborn as sns
from scipy.stats import entropy
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity as nx_modularity

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


from collections import defaultdict
import numpy as np
import networkx as nx

def build_service_assignment(graph, tau_assignment=0.0):
    """
    Build a hard assignment mapping consumer -> provider for one service graph at one month.
    Uses max 'assignment' weight among edges marked is_service_flow.

    Returns:
        assign: dict[consumer -> provider]
    """
    best = {}  # consumer -> (provider, weight)

    for u, v, data in graph.edges(data=True):
        if not data.get("is_service_flow", False):
            continue
        w = data.get("assignment", 0.0)
        if w is None or w <= tau_assignment:
            continue
        if u == v:
            continue

        # Assumption: edge u -> v is consumer -> provider
        consumer, provider = u, v

        cur = best.get(consumer)
        if cur is None or w > cur[1]:
            best[consumer] = (provider, float(w))

    assign = {c: p for c, (p, w) in best.items()}
    return assign


def assignment_to_communities(assign):
    """
    Convert consumer->provider mapping into provider->set(nodes) catchments.
    Includes the provider itself in its community set (optional but convenient for set ops).
    """
    comms = defaultdict(set)
    for consumer, provider in assign.items():
        comms[provider].add(consumer)
        comms[provider].add(provider)
    return comms


def labels_from_assignment(assign, all_nodes=None, missing_label=-1):
    """
    Build node->label mapping for NMI. For consumers, label is provider.
    Providers that have consumers can be labeled as themselves too.

    If all_nodes is provided, nodes without assignment get missing_label.
    """
    labels = {}
    for consumer, provider in assign.items():
        labels[consumer] = provider
        # optional: label provider as itself, to keep it in the label space
        labels.setdefault(provider, provider)

    if all_nodes is not None:
        for n in all_nodes:
            labels.setdefault(n, missing_label)

    return labels


def calculate_nmi(labels_t1, labels_t2):
    """
    Partition-consistent NMI between two node->label mappings.
    Assumes sklearn.metrics.normalized_mutual_info_score is used.
    """
    from sklearn.metrics import normalized_mutual_info_score

    common_nodes = sorted(set(labels_t1.keys()) & set(labels_t2.keys()))
    if not common_nodes:
        return np.nan

    y1 = [labels_t1[n] for n in common_nodes]
    y2 = [labels_t2[n] for n in common_nodes]
    return float(normalized_mutual_info_score(y1, y2))


def analyze_community_evolution(
    communities_t1: dict,
    communities_t2: dict,
    tau_match: float = 0.30,
    tau_persist: float = 0.80,
    growth_hi: float = 1.10,
    shrink_lo: float = 0.90
):
    """
    Community evolution for catchment-area communities (node sets).

    Returns keys:
    {'Persistence','Growth','Contraction','Split','Merge','Dissolution','Birth'}
    """
    evolution = {
        'Persistence': 0,
        'Growth': 0,
        'Contraction': 0,
        'Split': 0,
        'Merge': 0,
        'Dissolution': 0,
        'Birth': 0
    }

    t1_ids = list(communities_t1.keys())
    t2_ids = list(communities_t2.keys())

    adj_t1 = defaultdict(list)  # t1 -> [(t2, j)]
    adj_t2 = defaultdict(list)  # t2 -> [(t1, j)]

    for a in t1_ids:
        A = communities_t1[a]
        for b in t2_ids:
            B = communities_t2[b]
            union = A | B
            if not union:
                continue
            j = len(A & B) / len(union)
            if j >= tau_match:
                adj_t1[a].append((b, j))
                adj_t2[b].append((a, j))

    # Birth / Dissolution
    for a in t1_ids:
        if len(adj_t1[a]) == 0:
            evolution['Dissolution'] += 1
    for b in t2_ids:
        if len(adj_t2[b]) == 0:
            evolution['Birth'] += 1

    # Split / Merge (event count)
    for a in t1_ids:
        if len(adj_t1[a]) >= 2:
            evolution['Split'] += 1
    for b in t2_ids:
        if len(adj_t2[b]) >= 2:
            evolution['Merge'] += 1

    # Persistence/Growth/Contraction for mutual 1–1 stable matches
    used_t1, used_t2 = set(), set()
    for a in t1_ids:
        if len(adj_t1[a]) != 1:
            continue
        b, j = adj_t1[a][0]
        if len(adj_t2[b]) != 1:
            continue
        if j < tau_persist:
            continue
        if a in used_t1 or b in used_t2:
            continue

        used_t1.add(a)
        used_t2.add(b)

        size1 = len(communities_t1[a])
        size2 = len(communities_t2[b])

        if size2 > size1 * growth_hi:
            evolution['Growth'] += 1
        elif size2 < size1 * shrink_lo:
            evolution['Contraction'] += 1
        else:
            evolution['Persistence'] += 1

    return evolution

from collections import defaultdict
import numpy as np
from collections import defaultdict
import numpy as np

def calculate_temporal_metrics(all_results, settl_name, month_range, service_list, month_order):
    """
    Temporal metrics for catchment communities:
    monthly_communities[month][service][provider] = frozenset(nodes)

    No autarky. Modularity omitted (catchments).
    """

    # global service_list, month_order  # keep your current calling style

    # ---------- helpers ----------
    def _build_assignments_from_graph(graph):
        best = {}  # consumer -> (provider, w)
        for u, v, data in graph.edges(data=True):
            if not data.get("is_service_flow", False):
                continue
            w = data.get("assignment", 0)
            if w is None or w <= 0 or u == v:
                continue
            consumer, provider = u, v
            cur = best.get(consumer)
            if cur is None or w > cur[1]:
                best[consumer] = (provider, float(w))
        return {c: p for c, (p, _) in best.items()}

    def _labels_from_assignment(assign, nodes):
        labels = {n: -1 for n in nodes}
        for consumer, provider in assign.items():
            labels[consumer] = provider
            labels[provider] = provider
        return labels

    def _nmi(labels1, labels2):
        from sklearn.metrics import normalized_mutual_info_score
        common = sorted(set(labels1.keys()) & set(labels2.keys()))
        if not common:
            return np.nan
        y1 = [labels1[n] for n in common]
        y2 = [labels2[n] for n in common]
        return float(normalized_mutual_info_score(y1, y2))

    def _analyze_community_evolution_one_service(C1, C2,
                                                tau_match=0.30,
                                                tau_persist=0.80,
                                                growth_hi=1.10,
                                                shrink_lo=0.90):
        evo = {
            'Persistence': 0, 'Growth': 0, 'Contraction': 0,
            'Split': 0, 'Merge': 0, 'Dissolution': 0, 'Birth': 0
        }

        t1_ids = list(C1.keys())
        t2_ids = list(C2.keys())

        adj_t1 = defaultdict(list)
        adj_t2 = defaultdict(list)

        for a in t1_ids:
            A = C1[a]
            for b in t2_ids:
                B = C2[b]
                U = A | B
                if not U:
                    continue
                j = len(A & B) / len(U)
                if j >= tau_match:
                    adj_t1[a].append((b, j))
                    adj_t2[b].append((a, j))

        for a in t1_ids:
            if len(adj_t1[a]) == 0:
                evo['Dissolution'] += 1
        for b in t2_ids:
            if len(adj_t2[b]) == 0:
                evo['Birth'] += 1

        for a in t1_ids:
            if len(adj_t1[a]) >= 2:
                evo['Split'] += 1
        for b in t2_ids:
            if len(adj_t2[b]) >= 2:
                evo['Merge'] += 1

        used_t1, used_t2 = set(), set()
        for a in t1_ids:
            if len(adj_t1[a]) != 1:
                continue
            b, j = adj_t1[a][0]
            if len(adj_t2[b]) != 1:
                continue
            if j < tau_persist:
                continue
            if a in used_t1 or b in used_t2:
                continue

            used_t1.add(a)
            used_t2.add(b)

            size1 = len(C1[a])
            size2 = len(C2[b])

            if size2 > size1 * growth_hi:
                evo['Growth'] += 1
            elif size2 < size1 * shrink_lo:
                evo['Contraction'] += 1
            else:
                evo['Persistence'] += 1

        return evo

    def _freeze_service_comms(service_comms):
        """
        Convert provider->set into provider->frozenset, and plain dict (no defaultdict).
        """
        frozen = {}
        for provider, nodes in service_comms.items():
            frozen[provider] = frozenset(nodes)
        return frozen

    # ---------- outputs ----------
    temporal_metrics = {
        'jaccard_similarity': {},
        'nmi_scores': {},
        'persistence_coefficient': {},
        'modularity_evolution': {},   # intentionally empty
        'community_evolution': {}
    }

    monthly_communities = {}   # month -> service -> provider -> frozenset(nodes)
    monthly_assignments = {}   # month -> service -> consumer->provider
    monthly_nodes = {}         # month -> service -> set(nodes)

    # ---------- build monthly structures ----------
    for month_idx in month_range:
        per_service_assign = {}
        per_service_nodes = {}
        per_service_comms_tmp = defaultdict(lambda: defaultdict(set))

        for service in service_list:
            try:
                graph = all_results[settl_name][service]["stats"].graphs[month_idx]
            except Exception:
                continue

            assign = _build_assignments_from_graph(graph)
            per_service_assign[service] = assign
            per_service_nodes[service] = set(graph.nodes())

            for consumer, provider in assign.items():
                per_service_comms_tmp[service][provider].add(consumer)
                per_service_comms_tmp[service][provider].add(provider)

        # freeze communities NOW (critical)
        per_service_comms = {}
        for service, comms in per_service_comms_tmp.items():
            per_service_comms[service] = _freeze_service_comms(comms)

        monthly_assignments[month_idx] = per_service_assign
        monthly_nodes[month_idx] = per_service_nodes
        monthly_communities[month_idx] = per_service_comms

    months = sorted(monthly_communities.keys())

    # ---------- consecutive metrics ----------
    for i in range(len(months) - 1):
        m1, m2 = months[i], months[i + 1]
        key = f"{month_order[m1]}->{month_order[m2]}"

        services = sorted(set(monthly_assignments[m1].keys()) & set(monthly_assignments[m2].keys()))

        j_list, nmi_list, pers_list = [], [], []
        evo_sum = {
            'Persistence': 0, 'Growth': 0, 'Contraction': 0,
            'Split': 0, 'Merge': 0, 'Dissolution': 0, 'Birth': 0
        }

        # robust “diffs” check using frozen dicts
        diffs = []
        for s in services:
            if monthly_communities[m1][s] != monthly_communities[m2][s]:
                diffs.append(s)

        # DEBUG print (keep if you want)
        if key in {"Jan->Feb", "Feb->Mar", "Mar->Apr"}:
            print(f"[DEBUG] {key}: services compared={len(services)}, services with diffs={diffs}")

        for s in services:
            # 1) Jaccard on assignment pairs
            pairs1 = set(monthly_assignments[m1][s].items())
            pairs2 = set(monthly_assignments[m2][s].items())
            if pairs1 or pairs2:
                denom = len(pairs1 | pairs2)
                if denom > 0:
                    j_list.append(len(pairs1 & pairs2) / denom)

            # 2) NMI on node->label
            nodes = sorted(set(monthly_nodes[m1].get(s, set())) | set(monthly_nodes[m2].get(s, set())))
            if nodes:
                L1 = _labels_from_assignment(monthly_assignments[m1][s], nodes)
                L2 = _labels_from_assignment(monthly_assignments[m2][s], nodes)
                val = _nmi(L1, L2)
                if not np.isnan(val):
                    nmi_list.append(val)

            # 3) persistence coefficient (provider-consistent overlap)
            C1 = monthly_communities[m1][s]
            C2 = monthly_communities[m2][s]
            ps = []
            for provider, comm in C1.items():
                if provider in C2 and comm:
                    nxt = C2[provider]
                    if nxt:
                        ps.append(len(comm & nxt) / len(comm))
            if ps:
                pers_list.append(float(np.mean(ps)))

            # 4) evolution
            evo = _analyze_community_evolution_one_service(C1, C2)
            for k2, v2 in evo.items():
                evo_sum[k2] += int(v2)

        if j_list:
            temporal_metrics['jaccard_similarity'][key] = float(np.mean(j_list))
        if nmi_list:
            temporal_metrics['nmi_scores'][key] = float(np.mean(nmi_list))
        if pers_list:
            temporal_metrics['persistence_coefficient'][key] = float(np.mean(pers_list))

        # If no diffs, events other than Persistence must be zero
        if key in {"Jan->Feb", "Feb->Mar", "Mar->Apr"} and not diffs:
            try:
                assert evo_sum["Split"] == 0 and evo_sum["Merge"] == 0 \
                and evo_sum["Birth"] == 0 and evo_sum["Dissolution"] == 0 \
                and evo_sum["Growth"] == 0 and evo_sum["Contraction"] == 0, \
                f"Impossible events with identical communities for {key}: {evo_sum}"
            except AssertionError as e:
                print(f"[ERROR] {e}")

        temporal_metrics['community_evolution'][key] = evo_sum

        if key in {"Jan->Feb", "Feb->Mar", "Mar->Apr"}:
            print(f"[DEBUG] {key}: evo_sum={evo_sum}")

    return temporal_metrics, monthly_communities

# def calculate_temporal_metrics(
#     all_results,
#     settl_name,
#     month_range,
#     service_list,
#     month_order,
#     tau_assignment=0.0,
#     tau_match=0.30,
#     tau_persist=0.80,
#     compute_modularity=False,   # recommended False for catchments
# ):
#     """
#     Fixed temporal metrics for catchment-area (provider assignment) communities.
#     - No autarky.
#     - Partition-consistent NMI.
#     - Jaccard computed on assignment pairs (consumer->provider), not union of nodes.
#     - Community evolution computed on provider->set catchments.
#     """

#     temporal_metrics = {
#         'jaccard_similarity': {},         # Jaccard on assignment pairs
#         'nmi_scores': {},                 # NMI on node->label partitions
#         'persistence_coefficient': {},    # mean |C_t ∩ C_{t+1}| / |C_t| over matched providers
#         'modularity_evolution': {},       # optional, disabled by default
#         'community_evolution': {}
#     }

#     monthly_assignments = {}   # month -> (service -> dict[consumer->provider])
#     monthly_communities = {}   # month -> (service -> dict[provider->set(nodes)])
#     monthly_labelings = {}     # month -> (service -> dict[node->label])

#     months = list(month_range)

#     # ---- Build per-month, per-service assignments and communities ----
#     for month_idx in months:
#         per_service_assign = {}
#         per_service_comms = {}
#         per_service_labels = {}

#         for service in service_list:
#             try:
#                 graph = all_results[settl_name][service]["stats"].graphs[month_idx]
#             except Exception:
#                 continue

#             assign = build_service_assignment(graph, tau_assignment=tau_assignment)
#             comms = assignment_to_communities(assign)

#             # For NMI, use all nodes that appear in either side for that service graph
#             labels = labels_from_assignment(assign, all_nodes=list(graph.nodes()), missing_label=-1)

#             per_service_assign[service] = assign
#             per_service_comms[service] = comms
#             per_service_labels[service] = labels

#         monthly_assignments[month_idx] = per_service_assign
#         monthly_communities[month_idx] = per_service_comms
#         monthly_labelings[month_idx] = per_service_labels

#     months = sorted(monthly_assignments.keys())

#     # ---- Consecutive-month metrics, aggregated across services ----
#     for i in range(len(months) - 1):
#         m1, m2 = months[i], months[i + 1]
#         key = f"{month_order[m1]}->{month_order[m2]}"

#         per_service_j = []
#         per_service_nmi = []
#         per_service_persist = []
#         per_service_evo = []

#         # consider services present in both months
#         services = sorted(set(monthly_assignments[m1].keys()) & set(monthly_assignments[m2].keys()))

#         for service in services:
#             A1 = monthly_assignments[m1][service]
#             A2 = monthly_assignments[m2][service]

#             # 1) Jaccard on assignment pairs (consumer->provider)
#             pairs1 = set(A1.items())
#             pairs2 = set(A2.items())
#             if pairs1 or pairs2:
#                 j = len(pairs1 & pairs2) / len(pairs1 | pairs2) if (pairs1 | pairs2) else np.nan
#                 if not np.isnan(j):
#                     per_service_j.append(j)

#             # 2) NMI on node->label (partition-like; missing_label for unassigned nodes)
#             L1 = monthly_labelings[m1][service]
#             L2 = monthly_labelings[m2][service]
#             nmi = calculate_nmi(L1, L2)
#             if not np.isnan(nmi):
#                 per_service_nmi.append(nmi)

#             # 3) Persistence coefficient: provider-consistent overlap
#             C1 = monthly_communities[m1][service]
#             C2 = monthly_communities[m2][service]
#             pers_scores = []
#             for provider, comm in C1.items():
#                 if provider in C2 and comm:
#                     nxt = C2[provider]
#                     if nxt:
#                         pers_scores.append(len(comm & nxt) / len(comm))
#             if pers_scores:
#                 per_service_persist.append(float(np.mean(pers_scores)))

#             # 4) Community evolution (catchment sets)
#             evo = analyze_community_evolution(C1, C2, tau_match=tau_match, tau_persist=tau_persist)
#             per_service_evo.append(evo)

#         if per_service_j:
#             temporal_metrics['jaccard_similarity'][key] = float(np.mean(per_service_j))
#         if per_service_nmi:
#             temporal_metrics['nmi_scores'][key] = float(np.mean(per_service_nmi))
#         if per_service_persist:
#             temporal_metrics['persistence_coefficient'][key] = float(np.mean(per_service_persist))

#         # Aggregate evolution by summing counts across services
#         if per_service_evo:
#             evo_sum = {
#                 'Persistence': 0, 'Growth': 0, 'Contraction': 0,
#                 'Split': 0, 'Merge': 0, 'Dissolution': 0, 'Birth': 0
#             }
#             for evo in per_service_evo:
#                 for k2, v2 in evo.items():
#                     evo_sum[k2] += int(v2)
#             temporal_metrics['community_evolution'][key] = evo_sum

#     # ---- Optional modularity: disabled by default ----
#     # If you really want something here, we should define an appropriate graph
#     # and justify it in the paper. Otherwise leave it off.
#     if compute_modularity:
#         # placeholder: explicitly not implemented to avoid misleading results
#         # (You can re-enable later with a well-defined graph.)
#         pass

#     return temporal_metrics, monthly_communities

def calculate_modularity(G, partition, ):
    """
    Calculate modularity Q for a given partition
    """
    if G.number_of_edges() == 0:
        return 0
    
    # Convert node->community_id mapping into a list of community node sets
    communities_dict = defaultdict(set)
    for node, comm_id in partition.items():
        communities_dict[comm_id].add(node)
    communities = list(communities_dict.values())

    # Ensure we provide a true partition of G's node set:
    # - every node must appear in exactly one community
    # - add singleton communities for any nodes not present in `partition`
    covered_nodes = set().union(*communities) if communities else set()
    missing_nodes = set(G.nodes()) - covered_nodes
    for node in missing_nodes:
        communities.append({node})

    # Delegate to NetworkX's modularity implementation (unweighted)
    return nx_modularity(G, communities, weight=None)

def identify_stable_communities(all_results, settl_name, 
                               month_range, service_list, stability_threshold=0.5):
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

def compare_months(all_results, settl_name, service_list, month_order, month1=5, month2=8):
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



def quick_stability_check(all_results, settl_name, month_range, service_list, month_order):
    """Quick stability analysis only"""
    multi, temporal, super_stable = identify_stable_communities(all_results, settl_name, month_range, service_list, month_order)
    
    print(f"\n🎯 Stability Summary:")
    print(f"   Multi-service providers: {len(multi)}")
    print(f"   Temporally stable: {len(temporal)}")
    print(f"   Super-stable: {len(super_stable)}")
    
    if super_stable:
        best = max(super_stable.items(), key=lambda x: x[1]['stability_score'])
        print(f"   Best: {best[0]} (score={best[1]['stability_score']:.2f})")
    
    return multi, temporal, super_stable
