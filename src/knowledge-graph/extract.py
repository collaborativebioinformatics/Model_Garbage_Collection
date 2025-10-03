"""
Extract Backbone from Knowledge Graph on Disk

This script is a **simulates** output from a human interest query subgraph. It
creates a "backbone" by filtering triples from alzheimers_triples.csv to create
a connected subgraph with specific subjects, ensuring connectivity and size
constraints.
"""

import csv
from collections import defaultdict


def read_triples(file_path):
    """Read triples from CSV file."""
    triples = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            triples.append((row["subject"], row["predicate"], row["object"]))
    return triples


def build_graph_from_triples(triples):
    """Build a simple adjacency list graph from triples."""
    graph = defaultdict(set)
    for subject, predicate, obj in triples:
        graph[subject].add(obj)
        graph[obj].add(subject)  # Undirected graph
    return graph


def find_connected_components(graph):
    """Find connected components using DFS."""
    visited = set()
    components = []

    def dfs(node, component):
        if node in visited:
            return
        visited.add(node)
        component.add(node)
        for neighbor in graph[node]:
            dfs(neighbor, component)

    for node in graph:
        if node not in visited:
            component = set()
            dfs(node, component)
            components.append(component)

    return components


def is_connected(graph):
    """Check if the graph is connected."""
    if not graph:
        return True

    # Start DFS from any node
    start_node = next(iter(graph))
    visited = set()
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph[node] - visited)

    # Check if all nodes were visited
    all_nodes = set(graph.keys()) | set().union(*graph.values())
    return len(visited) == len(all_nodes)


def find_connected_subgraph(
    triples, target_subjects, max_objects_per_subject=5, min_rows=10, max_rows=30
):
    """
    Filter triples to create a connected subgraph with target subjects.

    Args:
        triples: List of (subject, predicate, object) tuples
        target_subjects: List of subject IDs to include
        max_objects_per_subject: Maximum connections per subject
        min_rows: Minimum number of triples to return
        max_rows: Maximum number of triples to return

    Returns:
        List of filtered triples
    """
    target_set = set(target_subjects)

    # Find all triples involving target subjects (as subject or object)
    relevant_triples = []
    for subject, predicate, obj in triples:
        if subject in target_set or obj in target_set:
            relevant_triples.append((subject, predicate, obj))

    print(f"Found {len(relevant_triples)} triples involving target subjects")

    # Build graph to check connectivity
    graph = build_graph_from_triples(relevant_triples)

    # Check which target subjects are in the graph
    all_nodes = set(graph.keys()) | set().union(*graph.values()) if graph else set()
    present_subjects = [s for s in target_subjects if s in all_nodes]
    print(
        f"Present target subjects: {len(present_subjects)} out of {len(target_subjects)}"
    )

    if len(present_subjects) < 2:
        print("Warning: Less than 2 target subjects found in the data")
        return []

    # Find the largest connected component containing target subjects
    connected_components = find_connected_components(graph)
    target_component = None
    max_target_count = 0

    for component in connected_components:
        target_count = len(set(present_subjects) & component)
        if target_count > max_target_count:
            max_target_count = target_count
            target_component = component

    print(f"Largest connected component has {max_target_count} target subjects")

    if target_component is None:
        print("No connected component found with target subjects")
        return []

    # Filter triples to only include those in the connected component
    component_triples = []
    for subject, predicate, obj in relevant_triples:
        if subject in target_component and obj in target_component:
            component_triples.append((subject, predicate, obj))

    # Limit objects per subject
    subject_object_count = defaultdict(int)
    filtered_triples = []

    for subject, predicate, obj in component_triples:
        if subject in target_set:  # Only limit for target subjects
            if subject_object_count[subject] < max_objects_per_subject:
                filtered_triples.append((subject, predicate, obj))
                subject_object_count[subject] += 1
        else:
            filtered_triples.append((subject, predicate, obj))

    # If we have too many triples, prioritize those connecting target subjects
    if len(filtered_triples) > max_rows:
        # Prioritize triples where both subject and object are target subjects
        priority_triples = [
            (s, p, o)
            for s, p, o in filtered_triples
            if s in target_set and o in target_set
        ]
        other_triples = [
            (s, p, o)
            for s, p, o in filtered_triples
            if not (s in target_set and o in target_set)
        ]

        # Take priority triples first, then fill with others
        final_triples = priority_triples[:max_rows]
        remaining_slots = max_rows - len(final_triples)
        if remaining_slots > 0:
            final_triples.extend(other_triples[:remaining_slots])
    else:
        final_triples = filtered_triples

    # If we have too few triples, try to add more from the broader network
    if len(final_triples) < min_rows:
        print(f"Only {len(final_triples)} triples found, trying to expand...")
        # Add more triples from the original relevant set
        additional_needed = min_rows - len(final_triples)
        current_subjects_objects = set()
        for s, p, o in final_triples:
            current_subjects_objects.add(s)
            current_subjects_objects.add(o)

        for subject, predicate, obj in relevant_triples:
            if len(final_triples) >= max_rows:
                break
            if (subject, predicate, obj) not in final_triples:
                if (
                    subject in current_subjects_objects
                    or obj in current_subjects_objects
                ):
                    final_triples.append((subject, predicate, obj))
                    current_subjects_objects.add(subject)
                    current_subjects_objects.add(obj)

    return final_triples


def main():
    """Main function to extract and filter the knowledge graph backbone."""
    # Target subjects to focus on
    target_subjects = [
        "MGI:5435319",
        "CHEBI:27881",
        "CHEBI:15940",
        "CHEBI:27470",
        "CHEBI:3962",
        "MONDO:0005148",
        "MONDO:0005266",
        "MONDO:0006626",
        "MONDO:0005016",
        "MONDO:0000960",
        "MONDO:0005147",
        "CHEBI:15365",
    ]

    input_files = [
        "outputs/antijoin_alzheimers_random.csv",
        "outputs/antijoin_alzheimers_llm.csv",
        "outputs/antijoin_alzheimers_llm_rag.csv",
    ]

    for input_file in input_files:
        output_file = input_file.replace(".csv", "-backbone.csv")
        print(f"Reading triples from {input_file}")
        triples = read_triples(input_file)
        print(f"Total triples loaded: {len(triples)}")

        print(f"Filtering for {len(target_subjects)} target subjects...")
        filtered_triples = find_connected_subgraph(
            triples,
            target_subjects,
            max_objects_per_subject=5,
            min_rows=10,
            max_rows=30,
        )

        if not filtered_triples:
            print("No suitable subgraph found!")
            return

        print(f"Final filtered triples: {len(filtered_triples)}")

        # Write filtered triples to output file
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "predicate", "object"])
            for subject, predicate, obj in filtered_triples:
                writer.writerow([subject, predicate, obj])

        print(f"Filtered triples written to {output_file}")

        # Print summary statistics
        subjects = set(s for s, p, o in filtered_triples)
        objects = set(o for s, p, o in filtered_triples)
        all_entities = subjects | objects
        target_entities_found = set(target_subjects) & all_entities

        print(f"\nSummary:")
        print(f"- Total triples: {len(filtered_triples)}")
        print(f"- Unique subjects: {len(subjects)}")
        print(f"- Unique objects: {len(objects)}")
        print(f"- Total unique entities: {len(all_entities)}")
        print(
            f"- Target subjects found: {len(target_entities_found)} out of {len(target_subjects)}"
        )
        print(f"- Target subjects in graph: {sorted(target_entities_found)}")

        # Check connectivity
        final_graph = build_graph_from_triples(filtered_triples)
        if is_connected(final_graph):
            print("- Graph is connected ✓")
        else:
            components = find_connected_components(final_graph)
            print(f"- Graph has {len(components)} connected components")


if __name__ == "__main__":
    main()
