# AI Image Deblurring Application

A powerful web application for removing blur and enhancing images using advanced AI-powered algorithms and traditional image processing techniques.

## 📋 Table of Contents

- [🚀 Quick Start](#quick-start)
- [📁 Features](#features)
- [🔧 Installation](#installation)
- [🎯 Usage Guide](#usage)
- [🤖 Enhancement Methods](#enhancement-methods)
- [🔌 API Documentation](#api-documentation)
- [🏗️ Architecture](#architecture)
- [🐛 Troubleshooting](#troubleshooting)
- [📚 License](#license)
- [🤝 Contributing](#contributing)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Modern web browser (Chrome, Firefox, Edge, Safari)
- 2GB+ RAM recommended
- 1GB+ disk space for temporary files

### Quick Launch
```bash
# Clone or download project
cd ai-image-deblurring

# Install dependencies
pip install -r requirements.txt

# Start the application
python app.py
```

### Access the Application
Open your web browser and navigate to:
```
http://localhost:5000
```

---
### Backend Technologies

- **Flask**: Web framework for Python
- **OpenCV**: Computer vision library for image processing
- **NumPy**: Numerical computing for array operations
- **Pillow**: Image processing library

### Frontend Technologies

- **HTML5**: Modern markup with semantic elements
- **TailwindCSS**: Utility-first CSS framework
- **Vanilla JavaScript**: No external dependencies
- **Font Awesome**: Icon library

### Image Processing Algorithms

The application implements several advanced deblurring techniques:

1. **Frequency Domain Filtering**: Wiener filter operates in the frequency domain
2. **Iterative Deconvolution**: Richardson-Lucy algorithm for blind deblurring
3. **Spatial Domain Enhancement**: Various sharpening and edge enhancement filters
4. **Adaptive Processing**: Local variance-based adaptive sharpening

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check if Python is installed correctly
   - Verify all dependencies are installed
   - Check if port 5000 is available

2. **Image processing fails**
   - Ensure the image is not corrupted
   - Try different deblurring methods
   - Adjust parameters for better results

3. **Slow processing**
   - Reduce kernel size and iterations
   - Use smaller images for testing
   - Close other applications using system resources

### Performance Tips

- Start with default parameters and adjust gradually
- Use the "Adaptive" method for general-purpose deblurring
- The "Combined" method gives best results but takes longer
- Smaller kernel sizes process faster

## License

This project is for educational and personal use. Please respect the licenses of the open-source libraries used.

## Contributing

Feel free to suggest improvements or report issues. Potential enhancements include:

- Batch processing for multiple images
- Additional deblurring algorithms
- Real-time preview of parameter changes
- Mobile app version

## Support

For questions or issues, please refer to the error messages in the browser console and server logs.
"# Image-Super-Resolution" 
"# Image-Super-Resolution" 
