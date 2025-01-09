import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from typing import Tuple, Callable

def load_image(image_path: str) -> Image.Image:
    try:
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            return None

        size_on_disk = os.path.getsize(image_path)

        with Image.open(image_path) as image:
            image_format = image.format
            width, height = image.size
            mode = image.mode  
            channels = len(image.getbands())  
    
            print(f"Image Metadata for '{os.path.basename(image_path)}':")
            print(f" - Format: {image_format}")
            print(f" - Size on Disk: {size_on_disk / 1024:.2f} KB")
            print(f" - Dimensions: {width}x{height} pixels")
            print(f" - Mode: {mode}")
            print(f" - Channels: {channels}")
    
            return image.copy()
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return None

def image_to_array(image: Image.Image) -> np.ndarray:
    try:
        image_array = np.array(image).astype(np.uint8)
        return image_array
    except Exception as e:
        print(f"Error converting image to array '{image}': {e}")
        return None

def array_to_image(image_array: np.ndarray, normalize: bool = True) -> Image.Image:
    try:
        if normalize and image_array.dtype != np.uint8:
            min_val = np.min(image_array)
            max_val = np.max(image_array)
            if max_val - min_val != 0:
                image_array = (255 * (image_array - min_val) / (max_val - min_val)).astype(np.uint8)
            else:
                image_array = np.zeros_like(image_array, dtype=np.uint8)
        
        image_array = np.clip(image_array, 0, 255)
        
        image = Image.fromarray(image_array)
        return image

    except Exception as e:
        print(f"Error converting array to image: {e}")
        return None

def apply(image, operation: Callable, return_array=False, **params):
    try:
        if isinstance(image, Image.Image):
            image_array = image_to_array(image)
        else:
            image_array = image_array
        
        result_image_array = operation(image_array, **params)
        
        if return_array == True:
            return result_image_array
        
        result_image = array_to_image(result_image_array)
        return result_image
    except Exception as e:
        print(f"Error applying operation {operation}: {e}")

def display_image(image: Image.Image, title: str = 'Image', figsize: Tuple[int, int] = (8, 6)) -> None:
    try:
        plt.figure(figsize=figsize)
        if image.mode == 'L': 
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()
        plt.close
    except Exception as e:
        print(f"Error displaying image: {e}")

def display_image_comparison(image_1: Image.Image, image_2: Image.Image, title_1: str = 'Image 1', title_2: str = 'Image 2', figsize: Tuple[int, int] = (16, 6)) -> None:
    try:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        ax = axes[0]
        if image_1.mode == 'L':
            ax.imshow(image_1, cmap='gray')
        else:
            ax.imshow(image_1)
        ax.set_title(title_1)
        ax.axis('off') 
        
        ax = axes[1]
        if image_2.mode == 'L':
            ax.imshow(image_2, cmap='gray')
        else:
            ax.imshow(image_2)
        ax.set_title(title_2)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"Error displaying image comparison: {e}")

def rgb_to_grayscale(I: np.ndarray) -> np.ndarray:
    M, N, D = I.shape
    I_grayscale = np.zeros((M, N, D), dtype=np.float64)
    
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    
    I_grayscale = np.dot(I[..., :3], weights)
    I_grayscale = np.clip(I_grayscale, 0, 255).astype(np.uint8)
    
    return I_grayscale

def generate_histogram(I: np.ndarray) -> np.ndarray:
    histogram, _ = np.histogram(I, bins=256, range=(0, 255))
    return histogram

def display_image_with_histogram(image: Image.Image, histogram: np.ndarray,  title: str = 'Image', hist_title: str = 'Histogram', figsize: Tuple[int, int] = (8, 10)) -> None:
    try:
        fig, (ax_image, ax_hist) = plt.subplots(
            2, 1, 
            figsize=figsize, 
            gridspec_kw={'height_ratios': [5, 1]},
            sharex=False
        )

        if image.mode == 'L':
            ax_image.imshow(image, cmap='gray')
        else:
            ax_image.imshow(image)
        ax_image.set_title(title)
        ax_image.axis('off')

        ax_hist.bar(range(len(histogram)), histogram, width=1, color='gray')
        ax_hist.set_title(hist_title)
        ax_hist.set_xlim([0, 255])
        ax_hist.set_xlabel('Intensity Value')
        ax_hist.set_ylabel('Pixel Count')

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    except Exception as e:
        print(f"Error displaying image and histogram: {e}")

def display_image_comparison_with_histogram(image_1: Image.Image, histogram_1: np.ndarray, image_2: Image.Image, histogram_2: np.ndarray, title_1: str = 'Image 1',
                                            hist_title_1: str = 'Histogram 1', title_2: str = 'Image 2', hist_title_2: str = 'Histogram 2', 
                                            figsize: Tuple[int, int] = (16, 10)) -> None:
    try:
        fig, axes = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]})
        
        ax = axes[0, 0]
        if image_1.mode == 'L':
            ax.imshow(image_1, cmap='gray')
        else:
            ax.imshow(image_1)
        ax.set_title(title_1)
        ax.axis('off') 
        
        ax = axes[0, 1]
        if image_2.mode == 'L':
            ax.imshow(image_2, cmap='gray')
        else:
            ax.imshow(image_2)
        ax.set_title(title_2)
        ax.axis('off')
        
        ax = axes[1, 0]
        ax.bar(range(len(histogram_1)), histogram_1, width=1, color='gray')
        ax.set_title(hist_title_1)
        ax.set_xlim([0, 255])
        ax.set_xlabel('Intensity Value')
        ax.set_ylabel('Pixel Count')
        
        ax = axes[1, 1]
        ax.bar(range(len(histogram_2)), histogram_2, width=1, color='gray')
        ax.set_title(hist_title_2)
        ax.set_xlim([0, 255])
        ax.set_xlabel('Intensity Value')
        ax.set_ylabel('Pixel Count')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    
    except Exception as e:
        print(f"Error displaying image comparison with histograms: {e}")

def match_histogram(I: np.ndarray, h: np.ndarray, h_prime: np.ndarray) -> np.ndarray:
    L = 256

    c = np.cumsum(h).astype(np.float64)
    c_norm = c / c[-1]

    c_prime = np.cumsum(h_prime).astype(np.float64)
    c_prime_norm = c_prime / c_prime[-1]

    T = np.interp(c_norm, c_prime_norm, np.arange(L))
    
    T = np.round(T).astype(np.uint8)
        
    I_prime = T[I]

    return I_prime

def equalize_histogram(h: np.ndarray) -> np.ndarray:
    L = 256

    c = np.cumsum(h).astype(np.float64)
    c_norm = c / c[-1]

    T = np.zeros(L, dtype=np.uint64)
    for r in range(L):
        T[r] = np.floor((L-1) * c_norm[r])

    h_eq = np.zeros(L, dtype=np.uint64)
    for k in range(L):
        h_eq[k] = np.sum(h[T == k])

    return h_eq
