import cv2
import numpy as np
import torch
import torch.nn as nn
import pyautogui
import mss
import time
import json
import sounddevice as sd
from PIL import Image
from pynput import mouse
import threading
import pygame

class AdDetectorNN(nn.Module):
    """
    Neural Network architecture for detecting advertisements in images.
    Uses a CNN with multiple convolutional layers followed by fully connected layers
    for binary classification (ad vs. non-ad).
    """
    def __init__(self):
        super(AdDetectorNN, self).__init__()
        # Convolutional layers for feature extraction
        # Input: 3 channels (RGB), Output: 16 feature maps
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Increase feature maps: 16 -> 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Final conv layer: 32 -> 64 feature maps
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers for classification
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Flattened conv output -> 512 neurons
        self.fc2 = nn.Linear(512, 2)  # Final classification layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        """
        Forward pass of the neural network.
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width)
        Returns:
            torch.Tensor: Probability distribution over classes (ad vs. non-ad)
        """
        x = self.pool(self.relu(self.conv1(x)))  # First conv block
        x = self.pool(self.relu(self.conv2(x)))  # Second conv block
        x = self.pool(self.relu(self.conv3(x)))  # Third conv block
        x = x.view(-1, 64 * 8 * 8)  # Flatten for fully connected layers
        x = self.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Output layer
        return torch.sigmoid(x)  # Apply sigmoid for probability output

class AdBlocker:
    """
    Main class for the OS-level ad blocking system.
    Handles screen capture, ad detection, overlay management, and audio control.
    """
    def __init__(self):
        # Initialize neural network model
        self.model = AdDetectorNN()
        # Set up GPU acceleration if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Load user configuration
        self.config = self.load_config()
        
        # Initialize screen capture utility
        self.sct = mss.mss()
        
        # Set up audio control
        pygame.mixer.init()
        
        # Initialize overlay system for ad blocking
        self.setup_overlay()
        
        # Control flag for the monitoring thread
        self.running = True
        
    def load_config(self):
        """
        Load configuration from JSON file or create default if not exists.
        Returns:
            dict: Configuration settings for the ad blocker
        """
        try:
            with open('ad_blocker_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default configuration
            default_config = {
                'block_color': (0, 0, 0),  # Default: black overlay
                'audio_action': 'mute',    # Options: 'mute', 'lower', 'none'
                'volume_reduction': 0.5,    # Volume multiplier when lowering
                'confidence_threshold': 0.8  # Minimum confidence for ad detection
            }
            # Save default configuration
            with open('ad_blocker_config.json', 'w') as f:
                json.dump(default_config, f)
            return default_config
    
    def setup_overlay(self):
        """
        Initialize the overlay window system for blocking ads.
        Creates a transparent, topmost window that can be positioned over ads.
        """
        self.root = tk.Tk()
        self.root.attributes('-alpha', 0.8)  # Set transparency
        self.root.attributes('-topmost', True)  # Keep window on top
        self.root.withdraw()  # Hide main window
        
        # Track active overlay windows
        self.active_overlays = {}

    def preprocess_image(self, image):
        """
        Prepare captured screen image for neural network processing.
        Args:
            image (numpy.ndarray): Raw screenshot image
        Returns:
            torch.Tensor: Preprocessed image tensor ready for the neural network
        """
        # Resize to expected input size
        image = cv2.resize(image, (64, 64))
        # Convert to PyTorch tensor and normalize
        image = torch.FloatTensor(image).permute(2, 0, 1)
        # Add batch dimension
        image = image.unsqueeze(0)
        return image.to(self.device)

    def detect_ads(self, frame):
        """
        Process a frame to detect advertisements.
        Args:
            frame (numpy.ndarray): Screenshot frame to analyze
        Returns:
            torch.Tensor: Binary tensor indicating presence of ads
        """
        preprocessed = self.preprocess_image(frame)
        with torch.no_grad():  # Disable gradient calculation for inference
            predictions = self.model(preprocessed)
        
        return predictions > self.config['confidence_threshold']

    def create_overlay(self, x, y, width, height):
        """
        Create a new overlay window to block an detected ad.
        Args:
            x, y (int): Screen coordinates for overlay placement
            width, height (int): Dimensions of the overlay
        Returns:
            tk.Toplevel: Configured overlay window
        """
        overlay = tk.Toplevel(self.root)
        # Position and size the overlay
        overlay.geometry(f'{width}x{height}+{x}+{y}')
        # Set overlay color from config
        overlay.configure(bg=f'#{self.config["block_color"][0]:02x}'
                          f'{self.config["block_color"][1]:02x}'
                          f'{self.config["block_color"][2]:02x}')
        overlay.attributes('-topmost', True)  # Keep overlay on top
        overlay.overrideredirect(True)  # Remove window decorations
        return overlay

    def handle_audio(self):
        """
        Manage audio settings when an ad is detected based on user configuration.
        """
        if self.config['audio_action'] == 'mute':
            pygame.mixer.music.set_volume(0)  # Completely mute
        elif self.config['audio_action'] == 'lower':
            # Reduce volume by configured amount
            pygame.mixer.music.set_volume(self.config['volume_reduction'])

    def monitor_screen(self):
        """
        Main monitoring loop that continuously checks for ads and updates overlays.
        Runs in a separate thread to maintain responsiveness.
        """
        while self.running:
            # Capture current screen
            screen = np.array(self.sct.grab(self.sct.monitors[0]))
            
            # Detect ads in current frame
            ad_regions = self.detect_ads(screen)
            
            if ad_regions.any():
                # Process each detected ad region
                for region in ad_regions:
                    x, y, w, h = region
                    overlay_key = f'{x}_{y}'
                    
                    # Create new overlay if needed
                    if overlay_key not in self.active_overlays:
                        self.active_overlays[overlay_key] = self.create_overlay(x, y, w, h)
                    
                # Handle audio adjustments
                self.handle_audio()
            
            # Clean up overlays for ads that are no longer present
            current_regions = {f'{r[0]}_{r[1]}' for r in ad_regions}
            for key in list(self.active_overlays.keys()):
                if key not in current_regions:
                    self.active_overlays[key].destroy()
                    del self.active_overlays[key]
            
            # Prevent excessive CPU usage
            time.sleep(0.1)

    def update_config(self, **kwargs):
        """
        Update configuration parameters and save to file.
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)
        with open('ad_blocker_config.json', 'w') as f:
            json.dump(self.config, f)

    def start(self):
        """
        Start the ad blocking system in a separate thread.
        """
        print("Starting OS-level ad blocker...")
        monitor_thread = threading.Thread(target=self.monitor_screen)
        monitor_thread.start()

    def stop(self):
        """
        Safely shut down the ad blocking system.
        """
        self.running = False
        self.root.destroy()
        pygame.mixer.quit()

def main():
    """
    Main entry point for the application.
    Initializes and runs the ad blocker until interrupted.
    """
    ad_blocker = AdBlocker()
    
    try:
        ad_blocker.start()
        
        # Example configuration update:
        # ad_blocker.update_config(block_color=(255, 0, 0))  # Change to red
        
        # Keep program running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping ad blocker...")
        ad_blocker.stop()

if __name__ == "__main__":
    main()