import torch
from torch.functional import F

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[Tensor])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    labels = torch.stack([
        labels > 0, # no hand / hand
        labels > 2, # left / right, only when labels > 0
        labels % 2 == 0 # no contact / contact 
    ], dim=1)

    hand_loss = F.binary_cross_entropy_with_logits(class_logits[:,0], labels[:,0])
    side_and_state_loss = F.binary_cross_entropy_with_logits(class_logits[sampled_pos_inds_subset][:,1:], labels[sampled_pos_inds_subset][:,1:])
    return hand_loss + side_and_state_loss, box_loss