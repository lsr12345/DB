import tensorflow as tf
import tensorflow.keras.backend as K


# 计算probability map的loss，返回取正负样本比例为1：3后的loss以及关于probability map的gt与pred的binary_crossentropy图
def balanced_crossentropy_loss(args, negative_ratio=3., scale=5.):
    pred, gt, mask = args
    pred = pred[..., 0]
    # 通过mask过滤无效点，得到有效positive mask图
    positive_mask = (gt * mask)
    # 同上， 得到有效negative mask图
    negative_mask = ((1 - gt) * mask)
    # 通过positive mask 计算 positive点数
    positive_count = tf.reduce_sum(positive_mask)
    # 同上，计算 获得符合正负样本比例为1：3的negative点数
    negative_count = tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio])
    # gt 与 pred 的二值交叉熵图
    loss = K.binary_crossentropy(gt, pred)
    # positive_loss
    positive_loss = loss * positive_mask
    # negative_loss
    negative_loss = loss * negative_mask
    # 正负样本比例为1：3 下的 negative_loss
    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))
    # 含OHEM的bceloss
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
    # 通过mask 过滤无用的 weights
    mask = mask * weights
    # 计算 pred 和 gt的融合值 ，得到交集值
    intersection = tf.reduce_sum(pred * gt * mask)
    # 分别计算 pred 和 gt的值相加，得到并集值
    union = tf.reduce_sum(pred * mask) + tf.reduce_sum(gt * mask) + 1e-6
    # 计算 交并loss
    loss = 1 - 2.0 * intersection / union
    return loss


# 计算thresh map的Loss：L1
def l1_loss(args, scale=10.):
    pred, gt, mask = args
    pred = pred[..., 0]
    # 计算有效 点数
    mask_sum = tf.reduce_sum(mask)
    # 计算 pred 和 gt的l1-loss，并用mask过滤
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
    
    # 1.预测的threshold map 2.实际threshold map 3.文字区域padded后的区域的标识图
    l1_loss_ = l1_loss([thresh, thresh_map, thresh_mask])
    # 1.预测的probability map 2.图片中文字区域经过shrinking（收缩）后的标识图 3.图片中有效区域的标识图（能够用来计算正负样本loss的区域）
    balanced_ce_loss_, dice_loss_weights = balanced_crossentropy_loss([binary, gt, mask])
    # 1. 预测的approximate binary map 2.图片中文字区域经过shrinking（收缩）后的标识图 3.图片中有效区域的标识图（能够用来计算正负样本loss的区域）
    dice_loss_ = dice_loss([thresh_binary, gt, mask, dice_loss_weights])
    return l1_loss_ + balanced_ce_loss_ + dice_loss_
