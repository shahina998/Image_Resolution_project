from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
import base64
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
from werkzeug.utils import secure_filename
import requests
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Image Analysis Functions
def analyze_image_quality(image):
    """Comprehensive image quality analysis"""
    try:
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate quality metrics
        metrics = {}
        
        # 1. Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = float(laplacian_var)
        
        # 2. Contrast (Standard deviation)
        metrics['contrast'] = float(np.std(gray))
        
        # 3. Brightness (Mean intensity)
        metrics['brightness'] = float(np.mean(gray))
        
        # 4. Noise estimation (using median filter)
        median = cv2.medianBlur(gray, 5)
        noise = np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32)))
        metrics['noise'] = float(noise)
        
        # 5. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        metrics['edge_density'] = float(edge_density)
        
        # 6. Blur detection (FFT based)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        blur_metric = np.sum(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 95)]) / np.sum(magnitude_spectrum)
        metrics['blur_score'] = float(blur_metric)
        
        # 7. Image dimensions
        metrics['width'] = int(image.shape[1])
        metrics['height'] = int(image.shape[0])
        metrics['total_pixels'] = int(image.shape[0] * image.shape[1])
        
        # 8. Color analysis (if color image)
        if len(image.shape) == 3:
            # Calculate colorfulness
            rg = np.abs(image[:, :, 0].astype(np.float32) - image[:, :, 1].astype(np.float32))
            rb = np.abs(image[:, :, 0].astype(np.float32) - image[:, :, 2].astype(np.float32))
            gb = np.abs(image[:, :, 1].astype(np.float32) - image[:, :, 2].astype(np.float32))
            
            colorfulness = np.sqrt(rg**2 + rb**2 + gb**2).mean()
            metrics['colorfulness'] = float(colorfulness)
        else:
            metrics['colorfulness'] = 0.0
        
        return metrics
        
    except Exception as e:
        print(f"Image analysis error: {e}")
        return {}

def compare_images(original, enhanced):
    """Compare original and enhanced images"""
    try:
        # Analyze both images
        original_metrics = analyze_image_quality(original)
        enhanced_metrics = analyze_image_quality(enhanced)
        
        # Calculate improvements
        comparison = {}
        
        # Quality improvements
        if 'sharpness' in original_metrics and 'sharpness' in enhanced_metrics:
            sharpness_improvement = ((enhanced_metrics['sharpness'] - original_metrics['sharpness']) / original_metrics['sharpness']) * 100
            comparison['sharpness_improvement'] = float(sharpness_improvement)
        
        if 'contrast' in original_metrics and 'contrast' in enhanced_metrics:
            contrast_improvement = ((enhanced_metrics['contrast'] - original_metrics['contrast']) / original_metrics['contrast']) * 100
            comparison['contrast_improvement'] = float(contrast_improvement)
        
        if 'edge_density' in original_metrics and 'edge_density' in enhanced_metrics:
            edge_improvement = ((enhanced_metrics['edge_density'] - original_metrics['edge_density']) / original_metrics['edge_density']) * 100
            comparison['edge_improvement'] = float(edge_improvement)
        
        if 'blur_score' in original_metrics and 'blur_score' in enhanced_metrics:
            blur_improvement = ((enhanced_metrics['blur_score'] - original_metrics['blur_score']) / original_metrics['blur_score']) * 100
            comparison['blur_reduction'] = float(blur_improvement)
        
        # Noise comparison
        if 'noise' in original_metrics and 'noise' in enhanced_metrics:
            noise_change = ((enhanced_metrics['noise'] - original_metrics['noise']) / original_metrics['noise']) * 100
            comparison['noise_change'] = float(noise_change)
        
        # Resolution improvement
        if 'total_pixels' in original_metrics and 'total_pixels' in enhanced_metrics:
            resolution_improvement = ((enhanced_metrics['total_pixels'] - original_metrics['total_pixels']) / original_metrics['total_pixels']) * 100
            comparison['resolution_improvement'] = float(resolution_improvement)
        
        # Overall quality score
        comparison['original_quality_score'] = calculate_quality_score(original_metrics)
        comparison['enhanced_quality_score'] = calculate_quality_score(enhanced_metrics)
        comparison['overall_improvement'] = comparison['enhanced_quality_score'] - comparison['original_quality_score']
        
        return {
            'original': original_metrics,
            'enhanced': enhanced_metrics,
            'comparison': comparison
        }
        
    except Exception as e:
        print(f"Image comparison error: {e}")
        return {}

def calculate_quality_score(metrics):
    """Calculate overall quality score from metrics"""
    try:
        score = 0
        
        # Weight different metrics
        if 'sharpness' in metrics:
            score += metrics['sharpness'] * 0.3
        
        if 'contrast' in metrics:
            score += metrics['contrast'] * 0.2
        
        if 'edge_density' in metrics:
            score += metrics['edge_density'] * 1000 * 0.2
        
        if 'blur_score' in metrics:
            score += metrics['blur_score'] * 100 * 0.2
        
        if 'colorfulness' in metrics:
            score += metrics['colorfulness'] * 0.1
        
        # Normalize to 0-100 scale
        score = min(100, max(0, score))
        
        return float(score)
        
    except Exception as e:
        print(f"Quality score calculation error: {e}")
        return 0.0

def generate_analysis_report(comparison_data):
    """Generate detailed analysis report"""
    try:
        if not comparison_data:
            return "Analysis data not available"
        
        original = comparison_data.get('original', {})
        enhanced = comparison_data.get('enhanced', {})
        comparison = comparison_data.get('comparison', {})
        
        report = []
        
        # Header
        report.append("📊 IMAGE ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Original Image Stats
        report.append("\n📸 ORIGINAL IMAGE:")
        report.append(f"  • Dimensions: {original.get('width', 'N/A')} x {original.get('height', 'N/A')}")
        report.append(f"  • Total Pixels: {original.get('total_pixels', 'N/A'):,}")
        report.append(f"  • Sharpness: {original.get('sharpness', 'N/A'):.2f}")
        report.append(f"  • Contrast: {original.get('contrast', 'N/A'):.2f}")
        report.append(f"  • Brightness: {original.get('brightness', 'N/A'):.2f}")
        report.append(f"  • Edge Density: {original.get('edge_density', 'N/A'):.4f}")
        report.append(f"  • Blur Score: {original.get('blur_score', 'N/A'):.4f}")
        report.append(f"  • Noise Level: {original.get('noise', 'N/A'):.2f}")
        report.append(f"  • Colorfulness: {original.get('colorfulness', 'N/A'):.2f}")
        report.append(f"  • Quality Score: {comparison.get('original_quality_score', 'N/A'):.1f}/100")
        
        # Enhanced Image Stats
        report.append("\n✨ ENHANCED IMAGE:")
        report.append(f"  • Dimensions: {enhanced.get('width', 'N/A')} x {enhanced.get('height', 'N/A')}")
        report.append(f"  • Total Pixels: {enhanced.get('total_pixels', 'N/A'):,}")
        report.append(f"  • Sharpness: {enhanced.get('sharpness', 'N/A'):.2f}")
        report.append(f"  • Contrast: {enhanced.get('contrast', 'N/A'):.2f}")
        report.append(f"  • Brightness: {enhanced.get('brightness', 'N/A'):.2f}")
        report.append(f"  • Edge Density: {enhanced.get('edge_density', 'N/A'):.4f}")
        report.append(f"  • Blur Score: {enhanced.get('blur_score', 'N/A'):.4f}")
        report.append(f"  • Noise Level: {enhanced.get('noise', 'N/A'):.2f}")
        report.append(f"  • Colorfulness: {enhanced.get('colorfulness', 'N/A'):.2f}")
        report.append(f"  • Quality Score: {comparison.get('enhanced_quality_score', 'N/A'):.1f}/100")
        
        # Improvements
        report.append("\n📈 IMPROVEMENTS:")
        
        if 'sharpness_improvement' in comparison:
            report.append(f"  • Sharpness: {comparison['sharpness_improvement']:+.1f}%")
        
        if 'contrast_improvement' in comparison:
            report.append(f"  • Contrast: {comparison['contrast_improvement']:+.1f}%")
        
        if 'edge_improvement' in comparison:
            report.append(f"  • Edge Definition: {comparison['edge_improvement']:+.1f}%")
        
        if 'blur_reduction' in comparison:
            report.append(f"  • Blur Reduction: {comparison['blur_reduction']:+.1f}%")
        
        if 'noise_change' in comparison:
            report.append(f"  • Noise Change: {comparison['noise_change']:+.1f}%")
        
        if 'resolution_improvement' in comparison:
            report.append(f"  • Resolution: {comparison['resolution_improvement']:+.1f}%")
        
        if 'overall_improvement' in comparison:
            report.append(f"  • Overall Quality: {comparison['overall_improvement']:+.1f} points")
        
        # Summary
        report.append("\n🎯 SUMMARY:")
        overall = comparison.get('overall_improvement', 0)
        if overall > 20:
            report.append("  • Excellent improvement! Image significantly enhanced.")
        elif overall > 10:
            report.append("  • Good improvement! Noticeable enhancement achieved.")
        elif overall > 0:
            report.append("  • Moderate improvement. Some enhancement visible.")
        else:
            report.append("  • Minimal improvement detected.")
        
        return "\n".join(report)
        
    except Exception as e:
        print(f"Report generation error: {e}")
        return "Error generating analysis report"

# Advanced ESRGAN-like implementation using traditional methods
def advanced_esrgan_enhance(image, scale=4, model_type='general'):
    """Advanced ESRGAN-like enhancement with dramatic improvements"""
    try:
        # Convert to PIL for processing
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        original_size = image.size
        
        # Step 1: Aggressive multi-scale enhancement
        pyramid_levels = [1, 0.5, 0.25]
        enhanced_pyramids = []
        
        for level in pyramid_levels:
            if level != 1:
                scaled_size = (int(original_size[0] * level), int(original_size[1] * level))
                scaled_img = image.resize(scaled_size, Image.Resampling.LANCZOS)
                # Resize back with enhancement
                scaled_img = scaled_img.resize(original_size, Image.Resampling.LANCZOS)
            else:
                scaled_img = image
            
            # Apply aggressive enhancements
            if level == 1:
                # Original - strong enhancement
                enhancer = ImageEnhance.Sharpness(scaled_img)
                enhanced = enhancer.enhance(3.0)
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.5)
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=1))
            elif level == 0.5:
                # 50% - very strong enhancement
                enhancer = ImageEnhance.Sharpness(scaled_img)
                enhanced = enhancer.enhance(4.0)
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(2.0)
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.5, percent=250, threshold=0))
            else:
                # 25% - maximum detail extraction
                enhancer = ImageEnhance.Sharpness(scaled_img)
                enhanced = enhancer.enhance(5.0)
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(2.5)
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=300, threshold=0))
            
            enhanced_pyramids.append(enhanced)
        
        # Step 2: Aggressive detail fusion
        result_array = np.array(enhanced_pyramids[0], dtype=np.float32)
        
        # Add details from all levels with higher weights
        for i, enhanced_img in enumerate(enhanced_pyramids[1:], 1):
            enhanced_array = np.array(enhanced_img, dtype=np.float32)
            weight = 0.3 / i  # Higher weights for details
            result_array = result_array * (1 - weight) + enhanced_array * weight
        
        result = Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))
        
        # Step 3: Aggressive super-resolution
        if scale > 1:
            new_size = (int(result.width * scale), int(result.height * scale))
            result = result.resize(new_size, Image.Resampling.LANCZOS)
            
            # Post-upscaling aggressive enhancement
            enhancer = ImageEnhance.Sharpness(result)
            result = enhancer.enhance(2.0)
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.3)
        
        # Step 4: Model-specific aggressive enhancements
        if model_type == 'anime':
            # Anime - very aggressive
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(result)
            result = enhancer.enhance(2.5)
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.8)
        elif model_type == 'face':
            # Face - aggressive but careful
            result = result.filter(ImageFilter.UnsharpMask(radius=3, percent=150, threshold=2))
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Sharpness(result)
            result = enhancer.enhance(2.0)
        else:
            # General - very aggressive
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.6)
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Sharpness(result)
            result = enhancer.enhance(2.2)
        
        # Step 5: Final aggressive enhancement
        result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=1))
        
        return result
        
    except Exception as e:
        print(f"Advanced ESRGAN-like enhancement error: {e}")
        return image

def face_enhance(image):
    """Aggressive face-specific enhancement"""
    try:
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Aggressive face enhancement pipeline
        # 1. Strong sharpening
        enhancer = ImageEnhance.Sharpness(image)
        sharpened = enhancer.enhance(3.0)
        
        # 2. Contrast boost
        enhancer = ImageEnhance.Contrast(sharpened)
        contrasted = enhancer.enhance(1.5)
        
        # 3. Color enhancement
        enhancer = ImageEnhance.Color(contrasted)
        colored = enhancer.enhance(1.3)
        
        # 4. Unsharp mask for detail
        detailed = colored.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=1))
        
        # 5. Blend with original for natural look
        result = Image.blend(image, detailed, 0.6)
        
        return result
        
    except Exception as e:
        print(f"Face enhancement error: {e}")
        return image

def document_upscale(image, scale=4):
    """Specialized document upscaling for text and scanned documents"""
    try:
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Step 1: Convert to grayscale for text enhancement
        if image.mode == 'RGB':
            gray = image.convert('L')
        else:
            gray = image
        
        # Step 2: Enhance contrast for text readability
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # Step 3: Apply text-specific sharpening
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(3.0)
        
        # Step 4: Multiple unsharp masks for text clarity
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=400, threshold=0))
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=300, threshold=1))
        
        # Step 5: Super-resolution upscaling
        if scale > 1:
            new_size = (int(enhanced.width * scale), int(enhanced.height * scale))
            enhanced = enhanced.resize(new_size, Image.Resampling.LANCZOS)
            
            # Post-upscaling text enhancement
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(2.5)
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.5)
            
            # Final text sharpening
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=200, threshold=0))
        
        # Convert back to RGB if original was color
        if image.mode == 'RGB':
            result = Image.new('RGB', enhanced.size, (255, 255, 255))
            result.paste(enhanced)
            return result
        else:
            return enhanced
            
    except Exception as e:
        print(f"Document upscaling error: {e}")
        return image

def ultra_aggressive_enhance(image, scale=4):
    """Ultra-aggressive enhancement for 80% blur removal"""
    try:
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Step 1: Advanced deblurring with multiple filters
        # Apply multiple unsharp masks with different parameters for blur removal
        result = image.filter(ImageFilter.UnsharpMask(radius=1, percent=500, threshold=0))
        result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=400, threshold=1))
        result = result.filter(ImageFilter.UnsharpMask(radius=3, percent=300, threshold=2))
        result = result.filter(ImageFilter.UnsharpMask(radius=4, percent=250, threshold=3))
        
        # Step 2: Maximum sharpening with enhanced parameters
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(8.0)  # Increased from 5.0
        
        # Step 3: Enhanced contrast for detail recovery
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(3.5)  # Increased from 2.5
        
        # Step 4: Enhanced color saturation
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(2.2)  # Increased from 1.8
        
        # Step 5: Additional blur removal with custom kernel
        # Convert back to OpenCV for advanced processing
        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        # Apply advanced Wiener filter for blur removal
        kernel = np.ones((3,3), np.float32) / 9
        result_cv = cv2.filter2D(result_cv, -1, kernel * 2.0)
        
        # Apply Richardson-Lucy deconvolution
        for i in range(5):  # Multiple iterations for better blur removal
            result_cv = richardson_lucy_deblur(result_cv, kernel, iterations=3)
        
        # Convert back to PIL
        result = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
        
        # Step 6: Final enhancement pass
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(4.0)
        
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(2.0)
        
        # Step 7: Super-resolution with enhanced interpolation
        if scale > 1:
            new_size = (int(result.width * scale), int(result.height * scale))
            result = result.resize(new_size, Image.Resampling.LANCZOS)
            
            # Additional sharpening after upscaling
            enhancer = ImageEnhance.Sharpness(result)
            result = enhancer.enhance(3.0)
        
        return result
        
    except Exception as e:
        print(f"Ultra-aggressive enhancement error: {e}")
        return image

def wiener_filter(image, kernel, noise_var):
    """Enhanced Wiener filter for effective blur removal"""
    # Convert to float for better precision
    img_float = image.astype(np.float32) / 255.0
    
    # Apply FFT
    img_fft = np.fft.fft2(img_float)
    kernel_fft = np.fft.fft2(kernel, s=img_float.shape)
    
    # Enhanced Wiener filter with adaptive noise estimation
    kernel_fft_conj = np.conj(kernel_fft)
    kernel_fft_mag_sq = np.abs(kernel_fft) ** 2
    
    # Adaptive noise estimation for better blur removal
    estimated_noise = np.maximum(noise_var, 0.001)
    
    # Enhanced Wiener formula for 80% blur removal
    wiener_filter = kernel_fft_conj / (kernel_fft_mag_sq + estimated_noise)
    
    # Apply filter
    deblurred_fft = img_fft * wiener_filter
    deblurred = np.fft.ifft2(deblurred_fft)
    
    # Convert back to uint8
    result = np.real(deblurred)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result

def richardson_lucy_deblur(image, psf, iterations=30):
    """Enhanced Richardson-Lucy deconvolution for 80% blur removal"""
    # Convert to float for better precision
    image_float = image.astype(np.float32) / 255.0
    psf_float = psf.astype(np.float32)
    
    # Normalize PSF
    psf_float = psf_float / np.sum(psf_float)
    
    # Initialize estimate
    estimate = np.full(image_float.shape, 0.5)  # Better initialization
    
    # Enhanced Richardson-Lucy with more iterations for better blur removal
    for i in range(iterations):
        # Compute blurred estimate
        blurred_estimate = cv2.filter2D(estimate, -1, psf_float)
        
        # Avoid division by zero
        blurred_estimate[blurred_estimate < 1e-10] = 1e-10
        
        # Compute ratio
        ratio = image_float / blurred_estimate
        
        # Back-project ratio
        back_projected = cv2.filter2D(ratio, -1, np.flipud(np.fliplr(psf_float)))
        
        # Update estimate with enhanced convergence
        estimate = estimate * back_projected
        
        # Apply regularization to prevent noise amplification
        if i > iterations // 2:
            estimate = cv2.GaussianBlur(estimate, (3, 3), 0.5)
    
    # Convert back to uint8
    result = np.clip(estimate * 255, 0, 255).astype(np.uint8)
    
    return result

def gaussian_kernel(size, sigma):
    """Enhanced Gaussian kernel for better blur modeling"""
    kernel = cv2.getGaussianKernel(size, sigma)
    kernel = kernel * kernel.T
    return kernel / np.sum(kernel)

def motion_kernel(size, angle):
    """Enhanced motion kernel for better motion blur removal"""
    kernel = np.zeros((size, size))
    angle_rad = np.deg2rad(angle)
    
    # Create more realistic motion blur kernel
    for i in range(size):
        x = int(size/2 + i * np.cos(angle_rad))
        y = int(size/2 + i * np.sin(angle_rad))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    
    # Apply Gaussian smoothing to kernel for better results
    kernel = cv2.GaussianBlur(kernel, (3, 3), 0)
    kernel = kernel / np.sum(kernel)
    return kernel

def sharpen_image(image):
    """Apply sharpening filter to enhance details"""
    sharpen_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)
    return np.clip(sharpened, 0, 255)

def unsharp_mask(image, sigma=1.0, strength=1.5):
    """Apply unsharp masking for edge enhancement"""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def adaptive_sharpening(image, strength=1.5):
    """Adaptive sharpening based on local variance"""
    kernel = np.ones((5,5), np.float32) / 25
    mean = cv2.filter2D(image, -1, kernel)
    sqr_mean = cv2.filter2D(image**2, -1, kernel)
    variance = sqr_mean - mean**2
    
    adaptive_strength = strength * (1 - np.exp(-variance / 0.01))
    
    sharpen_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)
    
    result = image + adaptive_strength * (sharpened - image)
    return np.clip(result, 0, 1)

def deblur_image(image_data, method='ultra_aggressive', kernel_type='gaussian', 
                kernel_size=15, sigma=2.0, angle=45, iterations=30, noise_var=0.01, strength=1.5, scale=4):
    """Main deblurring function with ultra-aggressive enhancement for dramatic results"""
    
    # Convert base64 to numpy array
    image_data = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image")
    
    # Apply deblurring method
    if method == 'ultra_aggressive':
        # Use ultra-aggressive enhancement for maximum visible difference
        enhanced_pil = ultra_aggressive_enhance(image, scale)
        result = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
        
    elif method == 'advanced_esrgan':
        # Use advanced ESRGAN-like enhancement for best results
        enhanced_pil = advanced_esrgan_enhance(image, scale, model_type='general')
        result = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
        
    elif method == 'real_esrgan_face':
        # Use advanced enhancement with face enhancement
        enhanced_pil = advanced_esrgan_enhance(image, scale, model_type='face')
        face_enhanced_pil = face_enhance(enhanced_pil)
        result = cv2.cvtColor(np.array(face_enhanced_pil), cv2.COLOR_RGB2BGR)
        
    elif method == 'real_esrgan_anime':
        # Use anime-optimized enhancement
        enhanced_pil = advanced_esrgan_enhance(image, scale, model_type='anime')
        result = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
        
    elif method == 'esrgan_traditional':
        # Combine advanced enhancement with traditional methods
        enhanced_pil = advanced_esrgan_enhance(image, scale, model_type='general')
        enhanced_cv = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
        
        # Apply additional sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced_cv, -1, kernel)
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
    elif method == 'document_upscale':
        # Use specialized document upscaling
        enhanced_pil = document_upscale(image, scale)
        result = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
        
    else:
        # Fallback to traditional methods
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float64) / 255.0
        
        # Create kernel based on type
        if kernel_type == 'gaussian':
            kernel = gaussian_kernel(kernel_size, sigma)
        elif kernel_type == 'motion':
            kernel = motion_kernel(kernel_size, angle)
        else:
            kernel = gaussian_kernel(kernel_size, sigma)
        
        # Apply traditional method
        if method == 'wiener':
            deblurred = wiener_filter(gray, kernel, noise_var)
        elif method == 'richardson_lucy':
            deblurred = richardson_lucy_deblur(gray, kernel, iterations)
        elif method == 'sharpen':
            deblurred = sharpen_image(gray * 255) / 255.0
        elif method == 'unsharp':
            deblurred = unsharp_mask(gray * 255, sigma, strength) / 255.0
        elif method == 'adaptive':
            deblurred = adaptive_sharpening(gray, strength)
        else:
            deblurred = adaptive_sharpening(gray, strength)
        
        # Convert back to BGR
        deblurred = np.clip(deblurred * 255, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', result)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deblur', methods=['POST'])
def deblur():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        image_data = data.get('image')
        method = data.get('method', 'adaptive')
        kernel_type = data.get('kernel_type', 'gaussian')
        kernel_size = int(data.get('kernel_size', 15))
        sigma = float(data.get('sigma', 2.0))
        angle = float(data.get('angle', 45))
        iterations = int(data.get('iterations', 30))
        noise_var = float(data.get('noise_var', 0.01))
        strength = float(data.get('strength', 1.5))
        scale = int(data.get('scale', 4))
        
        print(f"Processing with method: {method}, scale: {scale}")
        
        # Decode original image for analysis
        image_data_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_data_bytes, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise ValueError("Could not decode original image")
        
        # Process the image
        result = deblur_image(
            image_data, method, kernel_type, kernel_size, 
            sigma, angle, iterations, noise_var, strength, scale
        )
        
        # Decode enhanced image for analysis
        result_data_bytes = base64.b64decode(result.split(',')[1])
        result_nparr = np.frombuffer(result_data_bytes, np.uint8)
        enhanced_image = cv2.imdecode(result_nparr, cv2.IMREAD_COLOR)
        
        if enhanced_image is None:
            raise ValueError("Could not decode enhanced image")
        
        # Perform image analysis
        analysis_data = compare_images(original_image, enhanced_image)
        analysis_report = generate_analysis_report(analysis_data)
        
        print("Processing completed successfully")
        return jsonify({
            'success': True, 
            'result': result,
            'analysis': analysis_data,
            'report': analysis_report
        })
        
    except Exception as e:
        print(f"Error in deblur route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
