import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Import the modules we just built
from dsnn import build_dsnn
from losses import DualSupervisionLoss
from nc_fode import NCFODE_Enhancer

# Set random seed for reproducibility (as specified in Table 2)
tf.random.set_seed(42)
np.random.seed(42)

class Trainer:
    def __init__(self, dataset_path):
        self.batch_size = 8
        self.epochs = 50
        self.lr = 0.001
        self.image_size = (256, 256)
        
        # Initialize modules
        self.model = build_dsnn(input_shape=(256, 256, 1))
        self.optimizer = Adam(learning_rate=self.lr)
        self.loss_fn = DualSupervisionLoss(lambda_hat=0.2)
        self.ncfode = NCFODE_Enhancer(alpha=0.85)
        
        # Load datasets (assuming a function that loads your specific image paths)
        # self.train_ds = self.build_data_pipeline(dataset_path)

    def preprocess_image(self, image_path):
        """
        Pre-processing pipeline: Read, Resize, Median Filter, Min-Max Scale
        """
        # Read image
        img = cv2.imread(image_path.numpy().decode('utf-8'), cv2.IMREAD_GRAYSCALE)
        
        # Resize to 256x256
        img = cv2.resize(img, self.image_size)
        
        # Noise reduction: 3x3 Median Filter
        img = cv2.medianBlur(img, 3)
        
        # Min-Max Scaling [0, 1]
        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        
        return np.expand_dims(img, axis=-1)

    def augment_image(self, image):
        """
        Data augmentation: Flips and Rotations (+/- 15 degrees)
        """
        # Random horizontal and vertical flips
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Random rotation (approx +/- 15 degrees, which is ~0.26 radians)
        image = tf.keras.preprocessing.image.random_rotation(
            image.numpy(), 15, row_axis=0, col_axis=1, channel_axis=2
        )
        return image

    def tf_process_path(self, file_path):
        """Wrapper to run python/cv2 functions inside tf.data pipeline"""
        img = tf.py_function(self.preprocess_image, [file_path], tf.float32)
        img.set_shape([256, 256, 1])
        
        # Apply data augmentation
        img = tf.py_function(self.augment_image, [img], tf.float32)
        img.set_shape([256, 256, 1])
        return img

    def build_data_pipeline(self, file_paths):
        """Constructs the high-performance tf.data pipeline"""
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = dataset.map(self.tf_process_path, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    @tf.function
    def train_step(self, x_batch, target_masks, target_features):
        """
        Custom training step using GradientTape for Dual Supervision
        """
        with tf.GradientTape() as tape:
            # Forward pass through DSNN
            pred_fuse, pred_high = self.model(x_batch, training=True)
            
            # Calculate dual supervision loss
            total_loss, l_fuse, l_high = self.loss_fn.total_loss(
                target_masks, pred_fuse, 
                target_features, pred_high
            )

        # Backpropagation
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, l_fuse, l_high

    def train(self, file_paths):
        """Main training loop across epochs"""
        print("Starting training on RTX 3060 / 16GB RAM constraints...")
        train_ds = self.build_data_pipeline(file_paths)
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            epoch_loss_avg = tf.keras.metrics.Mean()
            
            for step, x_batch in enumerate(train_ds):
                
                # Note: In a full pipeline, you would generate target_masks and 
                # target_features (using the NC-FODE outputs as ground truth bounds).
                # For this boilerplate, we mock the targets.
                mock_target_masks = x_batch 
                mock_target_features = tf.zeros((tf.shape(x_batch)[0], 256, 256, 256))
                
                # Run optimization step
                loss, l_fuse, l_high = self.train_step(x_batch, mock_target_masks, mock_target_features)
                epoch_loss_avg.update_state(loss)
                
                if step % 10 == 0:
                    print(f"Step {step}: Total Loss = {loss:.4f} (Fuse: {l_fuse:.4f}, High: {l_high:.4f})")
            
            print(f"End of Epoch {epoch + 1} | Average Loss: {epoch_loss_avg.result():.4f}")

        print("Training complete. Model optimized.")
        self.model.save("nc_fode_dsnn_weights.h5")

