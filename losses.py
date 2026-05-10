import tensorflow as tf

class DualSupervisionLoss:
    def __init__(self, lambda_hat=0.2):
        """
        Initializes the custom loss class.
        lambda_hat: Balancing hyper-parameter for the high-level loss. 
                    The paper found 0.2 gives the best result.
        """
        self.lambda_hat = lambda_hat
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # Using MSE for the high-level feature comparison as implied by standard feature supervision
        self.mse = tf.keras.losses.MeanSquaredError()

    def dice_loss(self, y_true, y_pred, smooth=1e-6):
        """
        Computes the Dice loss for mask supervision.
        Equation 24 in the paper.
        """
        # Apply sigmoid to predictions if they are logits
        y_pred = tf.nn.sigmoid(y_pred)
        
        # Flatten the tensors
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        # Calculate Dice coefficient and subtract from 1
        dice_coef = (2. * intersection + smooth) / (union + smooth)
        return 1.0 - dice_coef

    def fuse_loss(self, y_true, y_pred):
        """
        Computes the fused loss (BCE + Dice) for the primary output mask.
        Equation 25 in the paper.
        """
        l_bce = self.bce(y_true, y_pred)
        l_dice = self.dice_loss(y_true, y_pred)
        
        return l_bce + l_dice

    def high_level_loss(self, features_true, features_pred):
        """
        Computes the loss for the high-level feature supervisor.
        """
        return self.mse(features_true, features_pred)

    def total_loss(self, y_true_mask, y_pred_mask, features_true, features_pred):
        """
        Computes the complete dual supervision total loss.
        Equation 22/26 in the paper.
        """
        l_fuse = self.fuse_loss(y_true_mask, y_pred_mask)
        l_high = self.high_level_loss(features_true, features_pred)
        
        l_total = l_fuse + (self.lambda_hat * l_high)
        
        return l_total, l_fuse, l_high