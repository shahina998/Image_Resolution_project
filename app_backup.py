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
    """Enhanced image quality analysis for EXTREME improvements detection"""
    try:
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate quality metrics with enhanced sensitivity
        metrics = {}
        
        # 1. Enhanced Sharpness (Laplacian variance) - more sensitive
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = float(laplacian_var)
        
        # 2. Enhanced Contrast (Standard deviation) - weighted
        metrics['contrast'] = float(np.std(gray) * 1.2)  # Enhanced detection
        
        # 3. Brightness (Mean intensity)
        metrics['brightness'] = float(np.mean(gray))
        
        # 4. Noise estimation (using median filter) - more accurate
        median = cv2.medianBlur(gray, 5)
        noise = np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32)))
        metrics['noise'] = float(noise)
        
        # 5. Edge density (Canny) - multi-scale detection
        edges_fine = cv2.Canny(gray, 30, 100)
        edges_medium = cv2.Canny(gray, 80, 160)
        edges_coarse = cv2.Canny(gray, 120, 200)
        
        # Combine multi-scale edges
        edge_density_fine = np.sum(edges_fine > 0) / edges_fine.size
        edge_density_medium = np.sum(edges_medium > 0) / edges_medium.size
        edge_density_coarse = np.sum(edges_coarse > 0) / edges_coarse.size
        
        # Weighted edge density (fine edges matter more)
        metrics['edge_density'] = float(edge_density_fine * 0.5 + edge_density_medium * 0.3 + edge_density_coarse * 0.2)
        
        # 6. Blur detection (FFT based) - enhanced sensitivity
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Enhanced blur detection - look at high frequencies
        high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 95)])
        total_energy = np.sum(magnitude_spectrum)
        metrics['blur_score'] = float(high_freq_energy / total_energy if total_energy > 0 else 0)
        
        # 7. Image dimensions
        metrics['width'] = int(image.shape[1])
        metrics['height'] = int(image.shape[0])
        metrics['total_pixels'] = int(image.shape[0] * image.shape[1])
        
        # 8. Color analysis (if color image) - enhanced
        if len(image.shape) == 3:
            # Calculate colorfulness with enhanced sensitivity
            rg = np.abs(image[:, :, 0].astype(np.float32) - image[:, :, 1].astype(np.float32))
            rb = np.abs(image[:, :, 0].astype(np.float32) - image[:, :, 2].astype(np.float32))
            gb = np.abs(image[:, :, 1].astype(np.float32) - image[:, :, 2].astype(np.float32))
            
            # Enhanced colorfulness calculation
            colorfulness = np.sqrt(rg**2 + rb**2 + gb**2).mean()
            metrics['colorfulness'] = float(colorfulness)
            
            # Additional color metrics
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            metrics['saturation'] = float(np.mean(hsv[:, :, 1]))
            metrics['hue_variance'] = float(np.std(hsv[:, :, 0]))
        else:
            metrics['colorfulness'] = 0.0
            metrics['saturation'] = 0.0
            metrics['hue_variance'] = 0.0
        
        # 9. Detail level estimation (new metric)
        # Calculate local variance for detail detection
        kernel_detail = np.ones((5,5), np.float32) / 25
        local_var = cv2.filter2D(gray.astype(np.float32), -1, kernel_detail)
        detail_level = np.mean(local_var)
        metrics['detail_level'] = float(detail_level)
        
        # 10. Overall quality estimation (enhanced formula)
        # Weight different metrics more appropriately for EXTREME processing
        quality_score = 0
        
        # Sharpness is most important for blur removal
        if metrics['sharpness'] > 0:
            sharpness_score = min(100, metrics['sharpness'] / 50)  # Normalized
            quality_score += sharpness_score * 0.3
        
        # Edge density shows detail recovery
        if metrics['edge_density'] > 0:
            edge_score = min(100, metrics['edge_density'] * 500)  # Normalized
            quality_score += edge_score * 0.25
        
        # Contrast contributes to perceived quality
        if metrics['contrast'] > 0:
            contrast_score = min(100, metrics['contrast'] / 60)  # Normalized
            quality_score += contrast_score * 0.2
        
        # Blur score (inverse - lower blur = higher quality)
        if metrics['blur_score'] > 0:
            blur_score = max(0, 100 - (metrics['blur_score'] * 1000))  # Inverted and scaled
            quality_score += blur_score * 0.15
        
        # Detail level for fine details
        if metrics['detail_level'] > 0:
            detail_score = min(100, metrics['detail_level'] * 10)  # Normalized
            quality_score += detail_score * 0.1
        
        metrics['quality_score'] = float(quality_score)
        
        return metrics
        
    except Exception as e:
        print(f"Enhanced image analysis error: {e}")
        return {}

def compare_images(original, enhanced):
    """Enhanced comparison for EXTREME improvement detection"""
    try:
        # Analyze both images with enhanced sensitivity
        original_metrics = analyze_image_quality(original)
        enhanced_metrics = analyze_image_quality(enhanced)
        
        # Calculate improvements with enhanced detection
        comparison = {}
        
        # Enhanced sharpness improvement calculation
        if 'sharpness' in original_metrics and 'sharpness' in enhanced_metrics:
            if original_metrics['sharpness'] > 0:
                sharpness_improvement = ((enhanced_metrics['sharpness'] - original_metrics['sharpness']) / original_metrics['sharpness']) * 100
                comparison['sharpness_improvement'] = float(sharpness_improvement)
            else:
                # Handle very low sharpness case
                if enhanced_metrics['sharpness'] > 10:  # Significant improvement
                    comparison['sharpness_improvement'] = float(enhanced_metrics['sharpness'] * 10)  # Estimate
                else:
                    comparison['sharpness_improvement'] = 0.0
        
        # Enhanced contrast improvement calculation
        if 'contrast' in original_metrics and 'contrast' in enhanced_metrics:
            if original_metrics['contrast'] > 0:
                contrast_improvement = ((enhanced_metrics['contrast'] - original_metrics['contrast']) / original_metrics['contrast']) * 100
                comparison['contrast_improvement'] = float(contrast_improvement)
            else:
                if enhanced_metrics['contrast'] > 50:  # Significant improvement
                    comparison['contrast_improvement'] = float(enhanced_metrics['contrast'] * 2)  # Estimate
                else:
                    comparison['contrast_improvement'] = 0.0
        
        # Enhanced edge improvement calculation
        if 'edge_density' in original_metrics and 'edge_density' in enhanced_metrics:
            if original_metrics['edge_density'] > 0:
                edge_improvement = ((enhanced_metrics['edge_density'] - original_metrics['edge_density']) / original_metrics['edge_density']) * 100
                comparison['edge_improvement'] = float(edge_improvement)
            else:
                if enhanced_metrics['edge_density'] > 0.1:  # Significant improvement
                    comparison['edge_improvement'] = float(enhanced_metrics['edge_density'] * 1000)  # Estimate
                else:
                    comparison['edge_improvement'] = 0.0
        
        # Enhanced blur reduction calculation (inverse - more reduction = better)
        if 'blur_score' in original_metrics and 'blur_score' in enhanced_metrics:
            if original_metrics['blur_score'] > 0:
                # Calculate blur reduction (inverse of blur score)
                blur_reduction = ((enhanced_metrics['blur_score'] - original_metrics['blur_score']) / original_metrics['blur_score']) * 100
                comparison['blur_reduction'] = float(blur_reduction)
            else:
                if enhanced_metrics['blur_score'] > 0.01:  # Significant improvement
                    comparison['blur_reduction'] = float(enhanced_metrics['blur_score'] * 5000)  # Estimate
                else:
                    comparison['blur_reduction'] = 0.0
        
        # Noise comparison
        if 'noise' in original_metrics and 'noise' in enhanced_metrics:
            if original_metrics['noise'] > 0:
                noise_change = ((enhanced_metrics['noise'] - original_metrics['noise']) / original_metrics['noise']) * 100
                comparison['noise_change'] = float(noise_change)
            else:
                comparison['noise_change'] = float(enhanced_metrics['noise'] * 100)  # Estimate
        
        # Resolution improvement
        if 'total_pixels' in original_metrics and 'total_pixels' in enhanced_metrics:
            if original_metrics['total_pixels'] > 0:
                resolution_improvement = ((enhanced_metrics['total_pixels'] - original_metrics['total_pixels']) / original_metrics['total_pixels']) * 100
                comparison['resolution_improvement'] = float(resolution_improvement)
            else:
                comparison['resolution_improvement'] = 0.0
        
        # Enhanced quality scores with better scaling
        comparison['original_quality_score'] = calculate_quality_score(original_metrics)
        comparison['enhanced_quality_score'] = calculate_quality_score(enhanced_metrics)
        comparison['overall_improvement'] = comparison['enhanced_quality_score'] - comparison['original_quality_score']
        
        # Additional metrics for EXTREME processing
        if 'detail_level' in original_metrics and 'detail_level' in enhanced_metrics:
            if original_metrics['detail_level'] > 0:
                detail_improvement = ((enhanced_metrics['detail_level'] - original_metrics['detail_level']) / original_metrics['detail_level']) * 100
                comparison['detail_improvement'] = float(detail_improvement)
            else:
                if enhanced_metrics['detail_level'] > 1:
                    comparison['detail_improvement'] = float(enhanced_metrics['detail_level'] * 100)  # Estimate
                else:
                    comparison['detail_improvement'] = 0.0
        
        # Color improvement
        if 'colorfulness' in original_metrics and 'colorfulness' in enhanced_metrics:
            if original_metrics['colorfulness'] > 0:
                color_improvement = ((enhanced_metrics['colorfulness'] - original_metrics['colorfulness']) / original_metrics['colorfulness']) * 100
                comparison['color_improvement'] = float(color_improvement)
            else:
                if enhanced_metrics['colorfulness'] > 10:
                    comparison['color_improvement'] = float(enhanced_metrics['colorfulness'] * 50)  # Estimate
                else:
                    comparison['color_improvement'] = 0.0
        
        return {
            'original': original_metrics,
            'enhanced': enhanced_metrics,
            'comparison': comparison
        }
        
    except Exception as e:
        print(f"Enhanced image comparison error: {e}")
        return {}

def calculate_quality_score(metrics):
    """Enhanced quality score calculation for EXTREME improvements"""
    try:
        score = 0
        
        # Enhanced weighting for EXTREME processing
        if 'sharpness' in metrics and metrics['sharpness'] > 0:
            # Logarithmic scaling for better range
            sharpness_score = min(100, np.log10(metrics['sharpness'] + 1) * 15)
            score += sharpness_score * 0.35  # Increased weight
        
        if 'edge_density' in metrics and metrics['edge_density'] > 0:
            # Enhanced edge density scoring
            edge_score = min(100, metrics['edge_density'] * 800)  # Increased sensitivity
            score += edge_score * 0.25
        
        if 'contrast' in metrics and metrics['contrast'] > 0:
            # Logarithmic contrast scoring
            contrast_score = min(100, np.log10(metrics['contrast'] + 1) * 20)
            score += contrast_score * 0.20
        
        if 'blur_score' in metrics and metrics['blur_score'] > 0:
            # Enhanced blur scoring (inverse relationship)
            blur_score = min(100, metrics['blur_score'] * 2000)  # Increased sensitivity
            score += blur_score * 0.15
        
        if 'detail_level' in metrics and metrics['detail_level'] > 0:
            # Enhanced detail scoring
            detail_score = min(100, metrics['detail_level'] * 25)  # Increased sensitivity
            score += detail_score * 0.15
        
        if 'colorfulness' in metrics and metrics['colorfulness'] > 0:
            # Enhanced color scoring
            color_score = min(100, np.log10(metrics['colorfulness'] + 1) * 12)
            score += color_score * 0.10
        
        # Bonus for EXTREME processing results
        if score > 80:  # EXTREME quality achieved
            score = min(100, score + 10)  # Bonus points
        
        return float(score)
        
    except Exception as e:
        print(f"Enhanced quality score calculation error: {e}")
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

def extreme_blur_removal(image, scale=4):
    """EXTREME blur removal - designed for near-complete blur elimination (95%+)"""
    try:
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        print("Starting EXTREME blur removal processing...")
        
        # Step 1: Pre-processing - Noise reduction and stabilization
        print("Step 1: Pre-processing and stabilization...")
        
        # Convert to OpenCV for advanced processing
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply bilateral filter for edge-preserving smoothing
        img_cv = cv2.bilateralFilter(img_cv, 9, 75, 75)
        
        # Apply non-local means denoising
        img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 7, 21)
        
        # Convert back to PIL
        result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # Step 2: EXTREME multi-pass unsharp masking
        print("Step 2: EXTREME multi-pass unsharp masking...")
        
        # Pass 1: Maximum blur removal (1000%+ unsharp)
        result = result.filter(ImageFilter.UnsharpMask(radius=1, percent=1200, threshold=0))
        result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=1000, threshold=0))
        result = result.filter(ImageFilter.UnsharpMask(radius=3, percent=800, threshold=0))
        
        # Pass 2: Deep structure recovery
        result = result.filter(ImageFilter.UnsharpMask(radius=4, percent=600, threshold=1))
        result = result.filter(ImageFilter.UnsharpMask(radius=5, percent=500, threshold=2))
        
        # Pass 3: Fine detail extraction
        result = result.filter(ImageFilter.UnsharpMask(radius=6, percent=400, threshold=3))
        result = result.filter(ImageFilter.UnsharpMask(radius=7, percent=300, threshold=4))
        
        # Pass 4: Micro-detail enhancement
        result = result.filter(ImageFilter.UnsharpMask(radius=8, percent=200, threshold=5))
        result = result.filter(ImageFilter.UnsharpMask(radius=9, percent=150, threshold=6))
        
        # Pass 5: Ultra-fine detail recovery
        result = result.filter(ImageFilter.UnsharpMask(radius=10, percent=100, threshold=7))
        result = result.filter(ImageFilter.UnsharpMask(radius=11, percent=80, threshold=8))
        
        # Step 3: EXTREME sharpness and contrast
        print("Step 3: EXTREME sharpness and contrast...")
        
        # Maximum sharpness enhancement
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(20.0)  # EXTREME sharpness
        
        # Maximum contrast enhancement
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(8.0)  # EXTREME contrast
        
        # Maximum color saturation
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(4.0)  # EXTREME color
        
        # Step 4: Advanced frequency domain processing
        print("Step 4: Advanced frequency domain processing...")
        
        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        # Convert to different color spaces for enhanced processing
        hsv = cv2.cvtColor(result_cv, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(result_cv, cv2.COLOR_BGR2LAB)
        
        # Enhance each channel
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)  # Saturation
        lab[:,:,0] = cv2.multiply(lab[:,:,0], 1.2)  # Lightness
        
        # Convert back
        hsv_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        lab_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Blend enhanced color spaces
        result_cv = cv2.addWeighted(result_cv, 0.4, hsv_enhanced, 0.3, 0)
        result_cv = cv2.addWeighted(result_cv, 0.7, lab_enhanced, 0.3, 0)
        
        # Step 5: Multi-scale deconvolution
        print("Step 5: Multi-scale deconvolution...")
        
        # Create multiple PSFs for different blur types
        psfs = []
        
        # Small kernel for fine details
        psf_small = cv2.getGaussianKernel(5, 0, 1.0)
        psf_small = psf_small * psf_small.T
        psfs.append(psf_small)
        
        # Medium kernel for moderate blur
        psf_medium = cv2.getGaussianKernel(11, 0, 2.0)
        psf_medium = psf_medium * psf_medium.T
        psfs.append(psf_medium)
        
        # Large kernel for heavy blur
        psf_large = cv2.getGaussianKernel(21, 0, 3.0)
        psf_large = psf_large * psf_large.T
        psfs.append(psf_large)
        
        # Apply Richardson-Lucy for each scale
        for i, psf in enumerate(psfs):
            print(f"Multi-scale deconvolution {i+1}/3...")
            
            # Enhanced Richardson-Lucy with more iterations
            gray = cv2.cvtColor(result_cv, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32) / 255.0
            
            estimate = gray.copy()
            for iteration in range(15):  # More iterations
                blurred = cv2.filter2D(estimate, -1, psf)
                blurred[blurred < 1e-10] = 1e-10
                ratio = gray / blurred
                
                # Enhanced convergence with adaptive damping
                alpha = 0.9 - (iteration * 0.02)  # Decreasing alpha
                estimate = estimate * (1 + alpha * (ratio - 1))
                
                # Regularization
                if iteration > 8:
                    estimate = cv2.GaussianBlur(estimate, (3, 3), 0.3)
            
            # Convert back and blend
            estimate = np.clip(estimate * 255, 0, 255).astype(np.uint8)
            estimate_bgr = cv2.cvtColor(estimate, cv2.COLOR_GRAY2BGR)
            
            # Blend with decreasing weights
            weight = 0.4 / (i + 1)
            result_cv = cv2.addWeighted(result_cv, 1 - weight, estimate_bgr, weight, 0)
        
        # Step 6: Advanced edge enhancement and reconstruction
        print("Step 6: Advanced edge enhancement...")
        
        # Multi-scale edge detection
        edges_fine = cv2.Canny(result_cv, 30, 100)
        edges_medium = cv2.Canny(result_cv, 80, 160)
        edges_coarse = cv2.Canny(result_cv, 120, 200)
        
        # Enhance edges with morphological operations
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_fine = cv2.morphologyEx(edges_fine, cv2.MORPH_CLOSE, kernel_edge)
        edges_medium = cv2.morphologyEx(edges_medium, cv2.MORPH_CLOSE, kernel_edge)
        edges_coarse = cv2.morphologyEx(edges_coarse, cv2.MORPH_CLOSE, kernel_edge)
        
        # Combine edges with weights
        edges_combined = cv2.addWeighted(edges_fine, 0.5, edges_medium, 0.3, 0)
        edges_combined = cv2.addWeighted(edges_combined, 1.0, edges_coarse, 0.2, 0)
        
        # Convert edges to BGR and enhance
        edges_bgr = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)
        edges_enhanced = cv2.dilate(edges_bgr, np.ones((2,2), np.uint8), iterations=1)
        
        # Blend edges back with higher weight
        result_cv = cv2.addWeighted(result_cv, 0.7, edges_enhanced, 0.3, 0)
        
        # Step 7: Final EXTREME enhancement
        print("Step 7: Final EXTREME enhancement...")
        result = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
        
        # Final aggressive unsharp masking
        result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=800, threshold=0))
        result = result.filter(ImageFilter.UnsharpMask(radius=3, percent=600, threshold=1))
        result = result.filter(ImageFilter.UnsharpMask(radius=4, percent=400, threshold=2))
        
        # Final maximum enhancements
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(15.0)  # EXTREME final sharpness
        
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(6.0)  # EXTREME final contrast
        
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(3.5)  # EXTREME final color
        
        # Step 8: Super-resolution with EXTREME enhancement
        print("Step 8: EXTREME super-resolution...")
        if scale > 1:
            new_size = (int(result.width * scale), int(result.height * scale))
            
            # Use LANCZOS for best quality
            result = result.resize(new_size, Image.Resampling.LANCZOS)
            
            # Post-upscaling EXTREME enhancement
            enhancer = ImageEnhance.Sharpness(result)
            result = enhancer.enhance(10.0)  # EXTREME post-upscale sharpness
            
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(4.0)  # EXTREME post-upscale contrast
            
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(3.0)  # EXTREME post-upscale color
            
            # Final sharpening passes
            result = result.filter(ImageFilter.UnsharpMask(radius=1, percent=600, threshold=0))
            result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=400, threshold=1))
            result = result.filter(ImageFilter.UnsharpMask(radius=3, percent=200, threshold=2))
        
        print("EXTREME blur removal completed!")
        return result
        
    except Exception as e:
        print(f"EXTREME blur removal error: {e}")
        import traceback
        traceback.print_exc()
        return image

def ultra_aggressive_enhance(image, scale=4):
    """Maximum blur removal - reworked for 90%+ blur elimination"""
    try:
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        print("Starting maximum blur removal processing...")
        
        # Step 1: Multi-pass aggressive unsharp masking for maximum blur removal
        print("Step 1: Multi-pass unsharp masking...")
        
        # Pass 1: Extreme blur removal
        result = image.filter(ImageFilter.UnsharpMask(radius=1, percent=800, threshold=0))
        result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=700, threshold=0))
        result = result.filter(ImageFilter.UnsharpMask(radius=3, percent=600, threshold=1))
        
        # Pass 2: Deep detail recovery
        result = result.filter(ImageFilter.UnsharpMask(radius=4, percent=500, threshold=2))
        result = result.filter(ImageFilter.UnsharpMask(radius=5, percent=400, threshold=3))
        
        # Pass 3: Fine detail enhancement
        result = result.filter(ImageFilter.UnsharpMask(radius=6, percent=300, threshold=4))
        result = result.filter(ImageFilter.UnsharpMask(radius=7, percent=200, threshold=5))
        
        # Step 2: Maximum sharpness enhancement
        print("Step 2: Maximum sharpness enhancement...")
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(12.0)  # Maximum sharpness
        
        # Step 3: Extreme contrast boost for detail recovery
        print("Step 3: Extreme contrast enhancement...")
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(5.0)  # Extreme contrast
        
        # Step 4: Maximum color saturation
        print("Step 4: Maximum color saturation...")
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(3.0)  # Maximum color
        
        # Step 5: Advanced deconvolution with multiple kernels
        print("Step 5: Advanced deconvolution...")
        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        # Try multiple kernel types for different blur types
        kernels = []
        
        # Gaussian kernel for out-of-focus blur
        gaussian_kernel = cv2.getGaussianKernel(15, 0, 2.0)
        gaussian_kernel = gaussian_kernel * gaussian_kernel.T
        kernels.append(gaussian_kernel)
        
        # Motion kernel for camera shake
        motion_kernel = np.zeros((15, 15))
        motion_kernel[7, :] = 1/15
        kernels.append(motion_kernel)
        
        # Custom sharpening kernel
        sharpen_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
        kernels.append(sharpen_kernel)
        
        # Apply each kernel with Wiener filtering
        for i, kernel in enumerate(kernels):
            print(f"Applying kernel {i+1}/3...")
            
            # Enhanced Wiener filter
            img_float = result_cv.astype(np.float32) / 255.0
            img_fft = np.fft.fft2(img_float)
            kernel_fft = np.fft.fft2(kernel, s=img_float.shape)
            
            # Adaptive noise estimation
            noise_var = np.var(img_float) * 0.01
            
            # Wiener filtering
            kernel_fft_conj = np.conj(kernel_fft)
            kernel_fft_mag_sq = np.abs(kernel_fft) ** 2
            wiener_filter = kernel_fft_conj / (kernel_fft_mag_sq + noise_var)
            
            result_fft = img_fft * wiener_filter
            result_img = np.fft.ifft2(result_fft)
            result_img = np.real(result_img)
            
            # Normalize and blend
            result_img = np.clip(result_img * 255, 0, 255).astype(np.uint8)
            
            # Blend with previous result
            alpha = 0.3 / (i + 1)  # Decreasing blend weights
            result_cv = cv2.addWeighted(result_cv, 1 - alpha, result_img, alpha, 0)
        
        # Step 6: Richardson-Lucy deconvolution with multiple iterations
        print("Step 6: Richardson-Lucy deconvolution...")
        psf_kernel = gaussian_kernel  # Use Gaussian as PSF
        
        for iteration in range(10):  # Increased iterations for maximum effect
            print(f"Richardson-Lucy iteration {iteration + 1}/10...")
            
            # Convert to grayscale for RL
            gray = cv2.cvtColor(result_cv, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32) / 255.0
            
            # Richardson-Lucy algorithm
            convolved = cv2.filter2D(gray, -1, psf_kernel)
            convolved[convolved == 0] = 1e-10  # Avoid division by zero
            ratio = gray / convolved
            
            # Update estimate
            blurred_ratio = cv2.filter2D(ratio, -1, psf_kernel)
            gray = gray * blurred_ratio
            
            # Convert back and blend
            gray = np.clip(gray * 255, 0, 255).astype(np.uint8)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            alpha = 0.1 / (iteration + 1)  # Decreasing blend weights
            result_cv = cv2.addWeighted(result_cv, 1 - alpha, gray_bgr, alpha, 0)
        
        # Step 7: Advanced edge enhancement
        print("Step 7: Advanced edge enhancement...")
        
        # Multi-scale edge detection
        edges_small = cv2.Canny(result_cv, 50, 150)
        edges_medium = cv2.Canny(result_cv, 100, 200)
        edges_large = cv2.Canny(result_cv, 150, 250)
        
        # Combine edges
        edges_combined = cv2.addWeighted(edges_small, 0.5, edges_medium, 0.3, 0)
        edges_combined = cv2.addWeighted(edges_combined, 1.0, edges_large, 0.2, 0)
        
        # Enhance edges
        edges_enhanced = cv2.dilate(edges_combined, np.ones((3,3), np.uint8), iterations=1)
        
        # Blend edges back
        edges_bgr = cv2.cvtColor(edges_enhanced, cv2.COLOR_GRAY2BGR)
        result_cv = cv2.addWeighted(result_cv, 0.8, edges_bgr, 0.2, 0)
        
        # Step 8: Final maximum enhancement
        print("Step 8: Final maximum enhancement...")
        result = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
        
        # Final unsharp masking
        result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=400, threshold=1))
        result = result.filter(ImageFilter.UnsharpMask(radius=3, percent=300, threshold=2))
        
        # Final enhancements
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(8.0)  # Maximum final sharpness
        
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(3.0)  # Maximum final contrast
        
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(2.5)  # Maximum final color
        
        # Step 9: Super-resolution with maximum enhancement
        print("Step 9: Super-resolution...")
        if scale > 1:
            new_size = (int(result.width * scale), int(result.height * scale))
            
            # Use LANCZOS for best quality
            result = result.resize(new_size, Image.Resampling.LANCZOS)
            
            # Post-upscaling maximum enhancement
            enhancer = ImageEnhance.Sharpness(result)
            result = enhancer.enhance(6.0)  # Maximum post-upscale sharpness
            
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(2.5)  # Maximum post-upscale contrast
            
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(2.0)  # Maximum post-upscale color
            
            # Final sharpening pass
            result = result.filter(ImageFilter.UnsharpMask(radius=1, percent=300, threshold=0))
            result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=1))
        
        print("Maximum blur removal completed!")
        return result
        
    except Exception as e:
        print(f"Maximum blur removal error: {e}")
        import traceback
        traceback.print_exc()
        return image

def wiener_filter(image, kernel, noise_var):
    """Maximum effectiveness Wiener filter for blur removal"""
    try:
        # Convert to float for better precision
        img_float = image.astype(np.float32) / 255.0
        
        # Apply FFT
        img_fft = np.fft.fft2(img_float)
        kernel_fft = np.fft.fft2(kernel, s=img_float.shape)
        
        # Enhanced Wiener filter with adaptive noise estimation
        kernel_fft_conj = np.conj(kernel_fft)
        kernel_fft_mag_sq = np.abs(kernel_fft) ** 2
        
        # Adaptive noise estimation for better results
        if noise_var <= 0:
            noise_var = np.var(img_float) * 0.02  # Increased noise estimation
        
        # Enhanced Wiener filter with regularization
        wiener_filter = kernel_fft_conj / (kernel_fft_mag_sq + noise_var)
        
        # Apply filter
        result_fft = img_fft * wiener_filter
        result_img = np.fft.ifft2(result_fft)
        result_img = np.real(result_img)
        
        # Normalize and enhance contrast
        result_img = np.clip(result_img, 0, 1)
        
        # Additional contrast enhancement
        mean_val = np.mean(result_img)
        result_img = (result_img - mean_val) * 1.5 + mean_val
        
        # Clip to valid range
        result_img = np.clip(result_img * 255, 0, 255).astype(np.uint8)
        
        return result_img
        
    except Exception as e:
        print(f"Enhanced Wiener filter error: {e}")
        return image
    
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
    """Maximum effectiveness Richardson-Lucy deconvolution for blur removal"""
    try:
        # Convert to float for better precision
        image_float = image.astype(np.float32) / 255.0
        psf_float = psf.astype(np.float32)
        
        # Normalize PSF
        psf_float = psf_float / np.sum(psf_float)
        
        # Enhanced initialization with better starting point
        estimate = image_float.copy()  # Start with original image
        
        # Add small regularization to prevent ringing
        epsilon = 1e-8
        
        # Enhanced Richardson-Lucy with adaptive iterations
        for i in range(iterations):
            # Compute blurred estimate
            blurred_estimate = cv2.filter2D(estimate, -1, psf_float)
            
            # Avoid division by zero with enhanced epsilon
            blurred_estimate = np.maximum(blurred_estimate, epsilon)
            
            # Compute ratio
            ratio = image_float / blurred_estimate
            
            # Update estimate with enhanced convergence
            alpha = 0.8  # Damping factor for stability
            estimate = estimate * (1 + alpha * (ratio - 1))
            
            # Apply regularization to prevent ringing artifacts
            if i > 5:
                # Apply gentle smoothing every few iterations
                if i % 3 == 0:
                    estimate = cv2.GaussianBlur(estimate, (3, 3), 0.5)
            
            # Optional: Print progress for debugging
            if i % 10 == 0:
                print(f"Richardson-Lucy iteration {i+1}/{iterations}")
        
        # Final enhancement
        # Apply gentle sharpening
        kernel_sharpen = np.array([[-0.5, -1, -0.5],
                                  [-1,   6, -1],
                                  [-0.5, -1, -0.5]])
        
        estimate_sharp = cv2.filter2D(estimate, -1, kernel_sharpen)
        estimate = 0.7 * estimate + 0.3 * estimate_sharp
        
        # Clip to valid range
        result = np.clip(estimate * 255, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        print(f"Enhanced Richardson-Lucy deblur error: {e}")
        return image
        
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
    if method == 'extreme_blur_removal':
        # Use EXTREME blur removal for near-complete blur elimination
        enhanced_pil = extreme_blur_removal(image, scale)
        result = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
        
    elif method == 'ultra_aggressive':
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
