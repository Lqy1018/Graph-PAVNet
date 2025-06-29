import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class HierarchicalGATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, max_hops=3):
        """
        Initialize the Hierarchical Graph Attention Network (GATNet).
        
        Args:
            in_channels (int): Number of input features for each node.
            out_channels (int): Number of output channels for the final prediction.
            max_hops (int): Maximum number of hops (graph layers) to consider in the hierarchy.
        """
        super(HierarchicalGATNet, self).__init__()
        self.max_hops = max_hops

        # First GAT layer: Processes 0-1 hop neighbors (node itself + max_hops info)
        self.conv1 = GATConv(in_channels + max_hops + 1, 8, heads=8, dropout=0.6)

        # Second GAT layer: Processes 0-2 hop neighbors (multiplied by number of heads)
        self.conv2 = GATConv(8 * 8, 16, heads=8, dropout=0.6)

        # Third GAT layer: Global propagation
        self.conv3 = GATConv(16 * 8, out_channels, heads=1, concat=False, dropout=0.6)

        # Hierarchical gating mechanism: weights for combining hop information
        self.gate_linear = torch.nn.Linear(16 * 8 + 1, 1)  # +1 for hop count information

    def forward(self, data):
        """
        Forward pass for the Hierarchical GAT Network.
        
        Args:
            data: A PyTorch Geometric data object containing:
                - x (Tensor): Node feature matrix of shape [N, F] (N = number of nodes, F = number of features).
                - edge_index (Tensor): Edge indices in COO format.
                - seed_mask (Tensor): Mask that identifies the seed nodes for hierarchy calculation.

        Returns:
            Tensor: Output predictions after applying GAT layers, with shape [N, out_channels].
        """
        x, edge_index, seed_mask = data.x, data.edge_index, data.seed_mask

        # Compute hierarchy mask and hop counts
        hierarchy_mask = self._compute_hierarchy_mask(edge_index, seed_mask)  # Shape: [N, (max_hops+1)]
        hop_counts = hierarchy_mask.argmax(dim=1).float()  # Shape: [N] (Hop level per node)

        hierarchy_mask = hierarchy_mask.to(x.device)
        hop_counts = hop_counts.to(x.device)

        # Feature enhancement: concatenate hierarchy mask with node features
        aug_x = torch.cat([x, hierarchy_mask], dim=1)  # Shape: [N, (F + (max_hops + 1))]

        # First layer propagation (0-1 hop nodes)
        h1 = self.conv1(aug_x, edge_index)  # Shape: [N, 64] (after convolution)
        h1 = F.elu(h1)  # Apply ELU activation
        h1 = F.dropout(h1, training=self.training)  # Apply dropout during training

        # Second layer propagation (0-2 hop nodes) + hierarchical gating
        h2 = self.conv2(h1, edge_index)  # Shape: [N, 128] (after convolution)
        gate_input = torch.cat([h2, hop_counts.unsqueeze(1)], dim=1)  # Shape: [N, (128 + 1)]
        gate = torch.sigmoid(self.gate_linear(gate_input))  # Shape: [N, 1] (sigmoid for gating)
        h2 = h2 * gate  # Apply gating mechanism (element-wise multiplication)

        # Third layer propagation (full graph propagation)
        h3 = self.conv3(h2, edge_index)  # Shape: [N, out_channels] (final output)

        return F.log_softmax(h3, dim=1)  # Apply log softmax for classification

    def _compute_hierarchy_mask(self, edge_index, seed_mask):
        """
        Compute the hierarchical mask for each node based on graph layers.
        
        Args:
            edge_index (Tensor): Edge indices in COO format.
            seed_mask (Tensor): Mask that identifies seed nodes for propagation.

        Returns:
            Tensor: Hierarchy mask of shape [N, (max_hops+1)], where each node has a one-hot encoding
                    for its hop level (0, 1, ..., max_hops).
        """
        layers = layerwise_propagation(edge_index, seed_mask)  # Propagate layers based on seed mask
        hierarchy_mask = torch.zeros((seed_mask.size(0), self.max_hops + 1))  # Shape: [N, (max_hops+1)]

        # Assign hierarchy levels to nodes
        for i, layer in enumerate(layers):
            if i > self.max_hops:
                break
            hierarchy_mask[layer, i] = 1  # Set the hierarchy mask for each node at the correct hop level

        return hierarchy_mask
