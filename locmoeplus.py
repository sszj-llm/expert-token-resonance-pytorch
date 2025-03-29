from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# An unofficial implementation of https://arxiv.org/abs/2406.00023 
# Expert-Token Resonance: Redefining MoE Expert Routing through affinity driven active selection.

class GrAPLayer(nn.Module):
    """Grouped Average Pooling as described in paper."""

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        assert (
            hidden_dim % num_experts == 0
        ), "Hidden dimension must be divisible by number of experts"
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.group_size = hidden_dim // num_experts

        # Initialize expert-token affinity matrix Waff as described in paper
        waff = torch.zeros(num_experts, hidden_dim)
        group_size = hidden_dim // num_experts
        scale = num_experts / hidden_dim  # n/d scaling factor

        for i in range(num_experts):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size
            waff[i, start_idx:end_idx] = scale

        self.register_buffer("waff", waff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute GrAP features using affinity matrix.
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_dim]
        Returns:
            GrAP features of shape [batch_size, seq_length, num_experts]
        """
        # Apply affinity matrix to get expert scores
        # waff has shape [num_experts, hidden_dim]
        # x has shape [batch_size, seq_length, hidden_dim]
        # Result will have shape [batch_size, seq_length, num_experts]
        return torch.matmul(x, self.waff.t())


class LocMoEPlusLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        expert_dim: int,
        min_capacity: int = 4,
        expert_dropout: float = 0.1,
        locality_weight: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.min_capacity = min_capacity
        self.locality_weight = locality_weight

        # GrAP layer for feature extraction with affinity matrix
        self.grap = GrAPLayer(input_dim, num_experts)

        # Initialize experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.GELU(),
                    nn.Dropout(expert_dropout),
                    nn.Linear(expert_dim, input_dim),
                )
                for _ in range(num_experts)
            ]
        )

        # Router uses GrAP features to generate scores
        self.router = nn.Linear(num_experts, num_experts, bias=False)
        nn.init.orthogonal_(self.router.weight)

        # Affinity threshold for adaptive capacity
        self.affinity_threshold = nn.Parameter(torch.tensor(0.5))

    def compute_affinity_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute expert-token affinity scores using equation (4) from paper.
        delta ti = cos(xt, wi) := xt^T wi/(||xt|| * ||wi||)

        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_dim]
        Returns:
            Affinity scores of shape [batch_size, seq_length, num_experts]
        """
        batch_size, seq_length, hidden_dim = x.shape

        # Compute x^T w
        numerator = torch.matmul(
            x, self.grap.waff.t()
        )  # [batch_size, seq_length, num_experts]

        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)  # [batch_size, seq_length, 1]

        w_norm = torch.norm(self.grap.waff, p=2, dim=-1)  # [num_experts]

        # Compute denominator ||x|| * ||w||
        denominator = torch.matmul(
            x_norm, w_norm.view(1, -1)
        )  # [batch_size, seq_length, num_experts]

        # Final affinity scores
        affinity_scores = numerator / (
            denominator + 1e-9
        )  # Added epsilon to avoid division by zero

        # cosine similarity is in [-1, 1]
        # verify this:
        # print(
        #    f"affinity_scores min: {affinity_scores.min()}, max: {affinity_scores.max()}"
        # )
        return affinity_scores

    def compute_adaptive_capacity(
        self, affinity_scores: torch.Tensor, sequence_length: int
    ) -> int:
        """Calculate adaptive capacity based on mean affinity score."""
        mean_affinity = affinity_scores.mean()
        adaptive_capacity = max(
            self.min_capacity,
            int(
                sequence_length * torch.sigmoid(mean_affinity - self.affinity_threshold)
            ),
        )
        return adaptive_capacity

    def route_tokens(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        local_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implement hybrid TCR+ECR routing with adaptive capacity.
        As described in equations (5) and (6) from paper.
        """
        batch_size, sequence_length, _ = inputs.shape

        # Compute affinity scores and routing probabilities
        affinity_scores = self.compute_affinity_scores(inputs)
        router_probs = F.softmax(affinity_scores, dim=-1)

        # Get adaptive capacity
        adaptive_capacity = self.compute_adaptive_capacity(
            affinity_scores, sequence_length
        )

        # TCR: Each token picks its top expert
        tcr_mask = torch.zeros(
            batch_size, sequence_length, self.num_experts, device=inputs.device
        )
        top_expert_idx = router_probs.argmax(dim=-1)

        batch_idx = torch.arange(batch_size, device=inputs.device).unsqueeze(1)
        seq_idx = torch.arange(sequence_length, device=inputs.device).unsqueeze(0)
        tcr_mask[batch_idx, seq_idx, top_expert_idx] = 1.0

        # ECR: Each expert picks its top-k tokens
        ecr_mask = torch.zeros_like(tcr_mask)
        for expert_idx in range(self.num_experts):
            expert_affinity = affinity_scores[..., expert_idx]
            top_k = min(adaptive_capacity, sequence_length)
            top_tokens = torch.topk(expert_affinity, k=top_k, dim=-1).indices

            batch_idx = torch.arange(batch_size, device=inputs.device).unsqueeze(1)
            ecr_mask[batch_idx, top_tokens, expert_idx] = 1.0

        # Combine masks for hybrid routing
        dispatch_mask = tcr_mask * ecr_mask

        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, self.num_experts)
            dispatch_mask = dispatch_mask * expanded_mask

        return router_probs, affinity_scores, dispatch_mask

    def compute_locality_loss(
        self, router_probs: torch.Tensor, local_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute KL divergence between current and localized distributions."""
        batch_size, seq_length, num_experts = router_probs.shape

        if local_indices is None:
            local_dist = torch.ones_like(router_probs) / num_experts
        else:
            local_dist = torch.zeros_like(router_probs)
            local_dist.scatter_(2, local_indices.unsqueeze(-1), 1.0)
            local_dist = local_dist / local_dist.sum(dim=-1, keepdim=True).clamp(
                min=1e-6
            )

        return F.kl_div(router_probs.log(), local_dist, reduction="batchmean")

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        local_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with expert-token resonance routing."""
        router_probs, affinity_scores, dispatch_mask = self.route_tokens(
            inputs, mask, local_indices
        )

        final_output = torch.zeros_like(inputs)
        for expert_idx, expert in enumerate(self.experts):
            expert_mask = dispatch_mask[..., expert_idx].unsqueeze(-1)
            if expert_mask.sum() > 0:
                expert_input = inputs * expert_mask
                expert_output = expert(expert_input)
                final_output = final_output + expert_output

        if self.training:
            # Router entropy loss
            router_entropy = (
                -(router_probs * torch.log(router_probs + 1e-9)).sum(-1).mean()
            )

            # Affinity loss
            affinity_loss = F.mse_loss(
                affinity_scores,
                torch.ones_like(affinity_scores) * self.affinity_threshold,
            )

            # Locality loss
            locality_loss = self.compute_locality_loss(router_probs, local_indices)

            # Combined auxiliary loss
            self.aux_loss = (
                router_entropy + affinity_loss + self.locality_weight * locality_loss
            )

        return final_output


def test_locmoe_plus():
    batch_size = 2
    sequence_length = 128
    input_dim = 768
    num_experts = 4
    expert_dim = 2048

    model = LocMoEPlusLayer(
        input_dim=input_dim,
        num_experts=num_experts,
        expert_dim=expert_dim,
    ).cuda()

    inputs = torch.randn(batch_size, sequence_length, input_dim).cuda()
    mask = torch.ones(batch_size, sequence_length).cuda()
    local_indices = torch.randint(0, num_experts, (batch_size, sequence_length)).cuda()

    outputs = model(inputs, mask, local_indices)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    if hasattr(model, "aux_loss"):
        print(f"Auxiliary loss: {model.aux_loss.item()}")


def test_locmoe_plus2():
    batch_size = 2
    sequence_length = 128
    input_dim = 768
    num_experts = 4
    expert_dim = 2048

    model = LocMoEPlusLayer(
        input_dim=input_dim,
        num_experts=num_experts,
        expert_dim=expert_dim,
    ).cuda()

    # Create sample inputs
    inputs = torch.randn(batch_size, sequence_length, input_dim).cuda()
    mask = torch.ones(batch_size, sequence_length).cuda()

    # We don't need local_indices for basic testing
    outputs = model(inputs, mask)

    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    if hasattr(model, "aux_loss"):
        print(f"Auxiliary loss: {model.aux_loss.item()}")


if __name__ == "__main__":
    test_locmoe_plus()
    test_locmoe_plus2()
