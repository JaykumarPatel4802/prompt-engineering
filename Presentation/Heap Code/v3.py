import heapq
import networkx as nx
import matplotlib.pyplot as plt

# Initialize a min-heap with elements 1 to 7
heap = [1, 2, 3, 4, 5, 6, 7]
heapq.heapify(heap)  # Ensuring it's a valid min-heap

def draw_heap(heap, title="Binary Min-Heap Visualization"):
    G = nx.DiGraph()
    
    # Function to assign nodes in a binary tree structure
    def add_edges(index):
        left = 2 * index + 1
        right = 2 * index + 2
        if left < len(heap):
            G.add_edge(heap[index], heap[left])
            add_edges(left)
        if right < len(heap):
            G.add_edge(heap[index], heap[right])
            add_edges(right)
    
    if heap:
        add_edges(0)  # Start from the root node (index 0)
    
    pos = hierarchy_pos(G, heap[0]) if heap else {}
    plt.figure(figsize=(5, 4))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', edge_color='gray')
    plt.title(title)
    plt.show()

def hierarchy_pos(G, root, width=2., vert_gap=1., xcenter=0.5, pos=None, parent=None, level=0):
    if pos is None:
        pos = {root: (xcenter, 0)}
    else:
        pos[root] = (xcenter, -level * vert_gap)
    
    children = list(G.successors(root))
    if children:
        dx = width / len(children)
        next_x = xcenter - width / 2 - dx / 2
        for child in children:
            next_x += dx
            pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, xcenter=next_x, pos=pos, parent=root, level=level+1)
    return pos

# Draw initial heap
draw_heap(heap, "Initial Heap")

# Step 1: Remove the root node (minimum value) and replace it with the last element
if heap:
    removed_node = heap[0]
    last_element = heap.pop()
    if heap:
        heap[0] = last_element  # Move last element to root
    draw_heap(heap, f"Heap after Root Replacement ({removed_node} -> {last_element})")

# Step 2: Heapify down to restore heap property
def heapify_down(heap, index=0):
    left = 2 * index + 1
    right = 2 * index + 2
    smallest = index
    
    if left < len(heap) and heap[left] < heap[smallest]:
        smallest = left
    if right < len(heap) and heap[right] < heap[smallest]:
        smallest = right
    
    if smallest != index:
        heap[smallest], heap[index] = heap[index], heap[smallest]  # Swap
        draw_heap(heap, f"Heap after Swap ({heap[index]} <-> {heap[smallest]})")
        heapify_down(heap, smallest)

if heap:
    heapify_down(heap)

# Final Heap
draw_heap(heap, "Final Heap after Heapify")
