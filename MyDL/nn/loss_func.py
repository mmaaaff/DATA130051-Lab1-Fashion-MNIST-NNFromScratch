from ..tensor import *
import numpy as np

def one_hot_encode(labels, num_classes):  # Note that this function always generates a 2D array
    if not isinstance(labels, MyTensor):
        raise TypeError("Labels must be a Mytensor")
    label_data = labels.data
    num_samples = 1 if labels.shape == (None,) else labels.shape[0]
    one_hot = np.zeros((num_samples, num_classes))
    one_hot[np.arange(num_samples), label_data] = 1
    encoded = MyTensor(one_hot, requires_grad=False)
    return encoded


class CrossEntropyLoss():
    def __init__(self):
        pass
    def calc_loss(self, pred, labels):
        eps = 1e-8
        if len(labels.shape) == 1:  # Labels are not one-hot encoded
            labels = one_hot_encode(labels, pred.shape[-1])  # 2D array
        if len(pred.shape) == 1:  # Pred is a single sample vector instead of a batch
            pred = pred.up_dim()
        # Tricks on calculating cross entropy loss(reduce calculation time and avoid overflow/underflow)
        pred_new = pred + 0  # Avoid changing the original pred
        pred_new.data[labels.data == 0] = 1
        if pred_new.requires_grad:
            pred_new.grad[labels.data == 0] = 0
        small_gap = MyTensor(np.zeros_like(pred_new.data), requires_grad=False)
        small_gap.data[pred_new.data < eps] = eps
        pred_new = pred_new + small_gap
        log_pred_new = log(pred_new)
        temp = log_pred_new * labels.neg()
        cross_entropy = temp.sum(axis=1)
        ave_loss = cross_entropy.sum() * (1 / pred_new.shape[0])
        ave_loss = ave_loss.item()
        return ave_loss
    def __call__(self, pred, labels):
        return self.calc_loss(pred, labels)