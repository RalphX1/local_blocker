# Neural Network Ad Blocker

A ad blocker that uses computer vision and neural networks to detect and block advertisements in real-time. Unlike traditional ad blockers that rely on URL filtering or element hiding, this solution uses visual recognition to identify and block ads directly on your screen.

## Features

- **Neural Network-Based Detection**: Uses a Convolutional Neural Network (CNN) to identify advertisements visually
- **Real-Time Screen Analysis**: Continuously monitors screen content for ads
- **Privacy**: Nothing is sent over the internet. Logs are saved locally
- **Visual Blocking**: Creates transparent overlays to hide detected advertisements
- **Smart Audio Management**: Automatically mutes or lowers volume during detected video ads
- **Customizable Settings**: Flexible configuration for blocking appearance and behavior
- **Resource Efficient**: Optimized for minimal system impact
- **GPU Acceleration**: Utilizes CUDA for faster processing when available

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for hardware acceleration)

### Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch
- OpenCV (cv2)
- NumPy
- PyAutoGUI
- MSS (screen capture)
- Pillow (PIL)
- Pynput
- Pygame
- Tkinter (usually comes with Python)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-ad-blocker.git
cd neural-ad-blocker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (optional):
```bash
python train_model.py --dataset path/to/dataset
```

## Usage

### Basic Usage

1. Start the ad blocker:
```bash
python ad_blocker.py
```

2. To stop the program:
- Press Ctrl+C in the terminal
- Or use the system tray icon (if implemented)

### Configuration

The ad blocker can be configured through `ad_blocker_config.json`:

```json
{
    "block_color": [0, 0, 0],     // RGB values for blocking overlay
    "audio_action": "mute",       // "mute", "lower", or "none"
    "volume_reduction": 0.5,      // Volume multiplier when lowering
    "confidence_threshold": 0.8    // Detection sensitivity (0-1)
}
```

### Programmatic Configuration

You can also update settings programmatically:

```python
ad_blocker.update_config(
    block_color=(255, 0, 0),  # Red blocking overlay
    audio_action='lower',     # Lower volume instead of muting
    volume_reduction=0.3      # Reduce volume to 30%
)
```

## How It Works

1. **Screen Capture**: Continuously captures screen content using MSS
2. **Neural Network Processing**:
   - Preprocesses captured frames
   - Runs them through a trained CNN
   - Identifies potential ad regions
3. **Overlay Management**:
   - Creates transparent windows to cover detected ads
   - Dynamically updates overlay positions
4. **Audio Control**:
   - Monitors for ad detection
   - Adjusts system volume based on configuration

## Neural Network Architecture

The ad detection model uses a CNN with the following structure:
- 3 Convolutional layers (3→16→32→64 channels)
- Max pooling layers
- 2 Fully connected layers
- Binary classification output (ad/no-ad)

## Training the Model

To train the model on your own dataset:

1. Prepare your dataset:
   - Collect screenshots of ads and non-ads
   - Organize into appropriate directories
   - Label the data accordingly

2. Run the training script:
```bash
python train_model.py --dataset path/to/dataset --epochs 100
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

### Areas for Improvement

- GUI for configuration
- System tray integration
- More sophisticated audio detection
- Additional neural network architectures
- Performance optimizations
- Multi-monitor support improvements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the PyTorch team for their excellent deep learning framework
- The MSS library for efficient screen capture
- The computer vision community for neural network architectures and techniques

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Submit a pull request
- Contact the maintainers

## Disclaimer

This software is for educational purposes only. Be aware of and respect website terms of service and local regulations regarding ad blocking. Some websites may not function properly with ad blocking enabled.

## Future Plans

- [ ] Implement system tray integration
- [ ] Add GUI for real-time configuration
- [ ] Improve detection accuracy
- [ ] Add support for custom neural network architectures
- [ ] Develop browser extension integration
- [ ] Create comprehensive documentation website
- [ ] Add automated testing suite