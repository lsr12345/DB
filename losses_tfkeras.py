import tensorflow as tf
import tensorflow.keras.backend as K


# 计算probability map的loss，返回取正负样本比例为1：3后的loss以及关于probability map的gt与pred的binary_crossentropy图
def balanced_crossentropy_loss(args, negative_ratio=3., scale=5.):
    pred, gt, mask = args
    pred = pred[..., 0]
    positive_mask = (gt * mask)

    negative_mask = ((1 - gt) * mask)

    positive_count = tf.reduce_sum(positive_mask)

    negative_count = tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio])

    loss = K.binary_crossentropy(gt, pred)
    # positive_loss
    positive_loss = loss * positive_mask
    # negative_loss
    negative_loss = loss * negative_mask

    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))
    balanced_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (
            positive_count + negative_count + 1e-6)
    balanced_loss = balanced_loss * scale
    return balanced_loss, loss


def dice_loss(args):
    """

    Args:
        pred: (b, h, w, 1)
        gt: (b, h, w)
        mask: (b, h, w)
        weights: (b, h, w)
    Returns:

    """
    pred, gt, mask, weights = args
    pred = pred[..., 0]
    # 归一化 weights
    weights = (weights - tf.reduce_min(weights)) / (tf.reduce_max(weights) - tf.reduce_min(weights)) + 1.
    mask = mask * weights
    intersection = tf.reduce_sum(pred * gt * mask)
    union = tf.reduce_sum(pred * mask) + tf.reduce_sum(gt * mask) + 1e-6
    loss = 1 - 2.0 * intersection / union
    return loss


# 计算thresh map的Loss：L1
def l1_loss(args, scale=10.):
    pred, gt, mask = args
    pred = pred[..., 0]
    mask_sum = tf.reduce_sum(mask)
    loss = K.switch(mask_sum > 0, tf.reduce_sum(tf.abs(pred - gt) * mask) / mask_sum, tf.constant(0.))
    loss = loss * scale
    return loss

# p, b_hat, gt_input, mask_input, t, thresh_input, thresh_mask_input
@tf.function
def db_loss(y_true, y_pred):
    
    binary = y_pred[..., 0]
#     print('binary.shape: ', binary.shape)
    thresh_binary = y_pred[..., 1]
#     print('thresh_binary.shape: ', thresh_binary.shape)
    gt = y_true[..., 0]
#     print('gt.shape: ', gt.shape)
    mask = y_true[..., 1]
#     print('mask.shape: ', mask.shape)
    thresh = y_pred[..., 2]
#     print('thresh.shape: ', thresh.shape)
    thresh_map = y_true[..., 2]
#     print('thresh_map.shape: ', thresh_map.shape)
    thresh_mask = y_true[..., 3]
#     print('thresh_mask.shape: ', thresh_mask.shape)
    
    l1_loss_ = l1_loss([thresh, thresh_map, thresh_mask])
    balanced_ce_loss_, dice_loss_weights = balanced_crossentropy_loss([binary, gt, mask])
    dice_loss_ = dice_loss([thresh_binary, gt, mask, dice_loss_weights])
    return l1_loss_ + balanced_ce_loss_ + dice_loss_
