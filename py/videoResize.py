import os
import numpy as np
from PIL import Image
import torch

class VideoResizeNode:
    """
    Node for resizing videos or image sequences with high-quality resampling methods.
    Supports both video input and image sequence input with professional resize options.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "resize_video"
    DESCRIPTION = "Resize video or image sequence to specified dimensions with multiple resampling options."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512, 
                    "min": 16, 
                    "max": 8192, 
                    "step": 1,
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 512, 
                    "min": 16, 
                    "max": 8192, 
                    "step": 1,
                    "display": "number"
                }),
                "interpolation": ([
                    "nearest",
                    "bilinear", 
                    "bicubic",
                    "area",
                    "lanczos"
                ], {"default": "lanczos"}),
                "method": ([
                    "stretch",
                    "keep proportion",
                    "fill / crop",
                    "pad"
                ], {"default": "keep proportion"}),
                "condition": ([
                    "always",
                    "downscale if bigger",
                    "upscale if smaller",
                    "if bigger area",
                    "if smaller area"
                ], {"default": "always"}),
                "pad_color": ("STRING", {
                    "default": "#000000"
                })
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "Video input from video nodes"}),
                "images": ("IMAGE", {"tooltip": "Image sequence input"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Frame",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

    def resize_video(self, width, height, interpolation="lanczos", method="keep proportion", 
                     condition="always", pad_color="#000000", video=None, images=None):
        """
        Resize video or image sequence with specified parameters.
        
        Args:
            width: Target width
            height: Target height  
            interpolation: Interpolation method
            method: Resize method (stretch, keep proportion, fill/crop, pad)
            condition: When to resize (always, conditionally)
            pad_color: Background color for padding
            video: Video input (image sequence from video nodes)
            images: Image sequence input
        """
        try:
            from ..dependency_manager import S4MotionLogger
        except:
            # Fallback logging
            class S4MotionLogger:
                @staticmethod
                def info(node, msg): print(f"[{node}] {msg}")
                @staticmethod
                def error(node, msg): print(f"[{node}] ERROR: {msg}")
                @staticmethod
                def success(node, msg): print(f"[{node}] SUCCESS: {msg}")
        
        # Determine input source
        input_tensor = None
        input_name = None
        
        if video is not None:
            input_tensor = video
            input_name = "video"
        elif images is not None:
            input_tensor = images
            input_name = "images"
        else:
            raise Exception("No input provided. Connect either 'video' or 'images' input.")
        
        S4MotionLogger.info("Video Resize", f"Processing {input_name} input...")
        
        # Convert input to frames
        if input_name == "video":
            # Handle VIDEO type input
            frames = self._load_video_frames(input_tensor)
        else:
            # Handle IMAGE type input (tensor)
            frames = self._tensor_to_pil_frames(input_tensor)
        
        if not frames:
            raise Exception("No frames found in input")
        
        S4MotionLogger.info("Video Resize", f"Processing {len(frames)} frames...")
        
        # Get original dimensions from first frame
        original_width, original_height = frames[0].size
        S4MotionLogger.info("Video Resize", f"Original dimensions: {original_width}x{original_height}")
        S4MotionLogger.info("Video Resize", f"Target dimensions: {width}x{height}")
        
        # Check resize condition
        if not self._should_resize(original_width, original_height, width, height, condition):
            S4MotionLogger.info("Video Resize", f"Condition '{condition}' not met - returning original frames")
            return (input_tensor,)
        
        # Parse pad color
        bg_color = self._parse_color(pad_color)
        
        # Get PIL interpolation method
        resample_method = self._get_interpolation_method(interpolation)
        
        S4MotionLogger.info("Video Resize", f"Using method: {method}, interpolation: {interpolation}")
        
        # Process frames
        resized_frames = []
        for i, frame in enumerate(frames):
            # Show progress for large sequences
            if len(frames) > 10 and (i % max(1, len(frames) // 10) == 0 or i == len(frames) - 1):
                progress = (i + 1) / len(frames) * 100
                S4MotionLogger.info("Video Resize", f"Progress: {progress:.1f}% ({i + 1}/{len(frames)})")
            
            resized_frame = self._smart_resize_frame(
                frame, width, height, method, resample_method, bg_color
            )
            resized_frames.append(resized_frame)
        
        # Convert back to ComfyUI format
        S4MotionLogger.info("Video Resize", "Converting frames to output format...")
        output_tensor = self._pil_frames_to_tensor(resized_frames)
        
        S4MotionLogger.success("Video Resize", f"Resize completed! Output shape: {output_tensor.shape}")
        
        return (output_tensor,)

    def _tensor_to_pil_frames(self, input_tensor):
        """Convert ComfyUI tensor to list of PIL Images"""
        frames = []
        
        # Handle torch tensor input
        if torch.is_tensor(input_tensor):
            # Convert to numpy
            if input_tensor.device != torch.device('cpu'):
                input_tensor = input_tensor.cpu()
            arr = input_tensor.numpy()
        else:
            arr = np.array(input_tensor)
        
        # ComfyUI format is typically (N, H, W, C) for image sequences
        if arr.ndim == 4:  # Batch of images: (N, H, W, C)
            for i in range(arr.shape[0]):
                frame_arr = arr[i]
                frame = self._array_to_pil(frame_arr)
                frames.append(frame)
                
        elif arr.ndim == 3:  # Single image: (H, W, C)
            frame = self._array_to_pil(arr)
            frames.append(frame)
        else:
            raise Exception(f"Unsupported input shape: {arr.shape}")
        
        return frames

    def _load_video_frames(self, video_obj):
        """Load frames from ComfyUI VideoFromFile object"""
        try:
            from ..dependency_manager import require_dependency
        except:
            # Fallback - check if OpenCV is available
            try:
                import cv2
            except ImportError:
                raise Exception("[Video Resize] OpenCV is required for video processing but not available")
        
        # Try to require OpenCV
        try:
            if not require_dependency('cv2', 'Video Processing', allow_fallback=False):
                raise Exception("[Video Resize] OpenCV is required for video processing but not available")
        except:
            # Fallback - assume OpenCV is available if we get here
            pass
        
        import cv2
        import tempfile
        import os
        
        try:
            from ..dependency_manager import S4MotionLogger
        except:
            class S4MotionLogger:
                @staticmethod
                def info(node, msg): print(f"[{node}] {msg}")
                @staticmethod
                def error(node, msg): print(f"[{node}] ERROR: {msg}")
        
        S4MotionLogger.info("Video Resize", "Processing VideoFromFile object")
        S4MotionLogger.info("Video Resize", f"Object type: {type(video_obj)}")
        
        try:
            # Try to get video stream source
            if hasattr(video_obj, 'get_stream_source'):
                stream_source = video_obj.get_stream_source()
                S4MotionLogger.info("Video Resize", f"Stream source: {stream_source}")
                
                # If stream source is a file path, use it directly
                if isinstance(stream_source, str) and os.path.exists(stream_source):
                    video_path = stream_source
                    S4MotionLogger.info("Video Resize", f"Using direct path: {video_path}")
                else:
                    # Save video to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                        temp_path = temp_file.name
                        S4MotionLogger.info("Video Resize", f"Saving video to temporary file: {temp_path}")
                        video_obj.save_to(temp_path)
                        video_path = temp_path
            else:
                # Fallback: save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                    S4MotionLogger.info("Video Resize", f"Saving video to temporary file: {temp_path}")
                    video_obj.save_to(temp_path)
                    video_path = temp_path
            
            S4MotionLogger.info("Video Resize", f"Loading video from: {video_path}")
            
            # Load video using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file: {video_path}")
            
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            S4MotionLogger.info("Video Resize", f"Total frames in video: {total_frames}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                from PIL import Image
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
                
                frame_count += 1
                if total_frames > 0 and frame_count % max(1, total_frames // 10) == 0:
                    S4MotionLogger.info("Video Resize", f"Loaded {frame_count}/{total_frames} frames")
            
            cap.release()
            
            # Clean up temporary file if we created one
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    S4MotionLogger.info("Video Resize", f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    S4MotionLogger.info("Video Resize", f"Warning: Could not clean up temporary file {temp_path}: {e}")
            
            S4MotionLogger.info("Video Resize", f"Successfully loaded {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            S4MotionLogger.error("Video Resize", f"Error processing video: {e}")
            
            # Try alternative approach: use get_components if available
            if hasattr(video_obj, 'get_components'):
                try:
                    S4MotionLogger.info("Video Resize", "Trying alternative approach with get_components")
                    components = video_obj.get_components()
                    S4MotionLogger.info("Video Resize", f"Components type: {type(components)}")
                    
                    # If components is already a list of frames, convert them
                    if isinstance(components, (list, tuple)):
                        frames = []
                        for i, component in enumerate(components):
                            if hasattr(component, 'shape') or isinstance(component, np.ndarray):
                                # Convert numpy array to PIL
                                if isinstance(component, np.ndarray):
                                    if component.dtype == np.float32 or component.dtype == np.float64:
                                        component = (component * 255).astype(np.uint8)
                                    from PIL import Image
                                    pil_frame = Image.fromarray(component)
                                    frames.append(pil_frame)
                            if i % 10 == 0:
                                S4MotionLogger.info("Video Resize", f"Processed component {i + 1}/{len(components)}")
                        
                        S4MotionLogger.info("Video Resize", f"Successfully loaded {len(frames)} frames from components")
                        return frames
                    
                except Exception as comp_e:
                    S4MotionLogger.error("Video Resize", f"Components approach failed: {comp_e}")
            
            raise Exception(f"Failed to load video frames: {e}")

    def _array_to_pil(self, arr):
        """Convert numpy array to PIL Image"""
        # Ensure proper data type
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            # Assume values are in [0, 1] range
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).round().astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        
        # Handle different channel counts
        if arr.ndim == 2:
            # Grayscale
            return Image.fromarray(arr, mode='L')
        elif arr.shape[2] == 1:
            # Grayscale with channel dimension
            return Image.fromarray(arr.squeeze(2), mode='L')
        elif arr.shape[2] == 3:
            # RGB
            return Image.fromarray(arr, mode='RGB')
        elif arr.shape[2] == 4:
            # RGBA
            return Image.fromarray(arr, mode='RGBA')
        else:
            raise Exception(f"Unsupported channel count: {arr.shape[2]}")

    def _parse_color(self, color_str):
        """Parse color string to RGB tuple"""
        try:
            if color_str.startswith('#'):
                color_str = color_str[1:]
            
            if len(color_str) == 6:
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
                return (r, g, b)
            else:
                # Default to black
                return (0, 0, 0)
        except:
            # Default to black on parse error
            return (0, 0, 0)

    def _get_interpolation_method(self, method_name):
        """Get PIL interpolation method from string"""
        method_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "area": getattr(Image, 'BOX', Image.BILINEAR),  # BOX is similar to area interpolation
            "lanczos": Image.LANCZOS,
        }
        return method_map.get(method_name, Image.LANCZOS)
    
    def _should_resize(self, orig_w, orig_h, target_w, target_h, condition):
        """Check if resize should be performed based on condition"""
        if condition == "always":
            return True
        elif condition == "downscale if bigger":
            return orig_w > target_w or orig_h > target_h
        elif condition == "upscale if smaller":
            return orig_w < target_w or orig_h < target_h
        elif condition == "if bigger area":
            return (orig_w * orig_h) > (target_w * target_h)
        elif condition == "if smaller area":
            return (orig_w * orig_h) < (target_w * target_h)
        return True

    def _smart_resize_frame(self, frame, width, height, method, interpolation, bg_color):
        """Smart resize frame based on method"""
        orig_w, orig_h = frame.size
        
        if method == "stretch":
            # Direct stretch to target size
            return frame.resize((width, height), interpolation)
        
        elif method == "keep proportion":
            # Maintain aspect ratio, fit within target dimensions (letterbox)
            aspect_ratio = orig_w / orig_h
            target_aspect = width / height
            
            if aspect_ratio > target_aspect:
                # Image is wider, fit to width
                new_w = width
                new_h = int(width / aspect_ratio)
            else:
                # Image is taller, fit to height
                new_h = height
                new_w = int(height * aspect_ratio)
            
            # Resize to calculated dimensions
            resized = frame.resize((new_w, new_h), interpolation)
            
            # Create canvas with background color
            if frame.mode == 'RGBA':
                canvas = Image.new('RGBA', (width, height), bg_color + (255,))
            else:
                canvas = Image.new('RGB', (width, height), bg_color)
            
            # Paste resized image centered
            x = (width - new_w) // 2
            y = (height - new_h) // 2
            
            if resized.mode == 'RGBA':
                canvas.paste(resized, (x, y), resized)
            else:
                canvas.paste(resized, (x, y))
            
            return canvas
        
        elif method == "fill / crop":
            # Fill target dimensions, crop excess
            aspect_ratio = orig_w / orig_h
            target_aspect = width / height
            
            if aspect_ratio > target_aspect:
                # Image is wider, fit to height and crop width
                new_h = height
                new_w = int(height * aspect_ratio)
            else:
                # Image is taller, fit to width and crop height
                new_w = width
                new_h = int(width / aspect_ratio)
            
            # Resize to calculated dimensions
            resized = frame.resize((new_w, new_h), interpolation)
            
            # Crop from center to target dimensions
            left = (new_w - width) // 2
            top = (new_h - height) // 2
            right = left + width
            bottom = top + height
            
            return resized.crop((left, top, right, bottom))
        
        elif method == "pad":
            # Same as keep proportion but with padding
            return self._smart_resize_frame(frame, width, height, "keep proportion", interpolation, bg_color)
        
        else:
            # Default to stretch
            return frame.resize((width, height), interpolation)


    def _pil_frames_to_tensor(self, frames):
        """Convert PIL frames back to ComfyUI tensor format"""
        frame_arrays = []
        
        for frame in frames:
            # Convert to numpy array
            arr = np.array(frame)
            
            # Ensure we have 3D array (H, W, C)
            if arr.ndim == 2:
                arr = arr[..., None]  # Add channel dimension for grayscale
            
            # Convert to float32 [0,1] range
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            
            # Ensure RGB (3 channels) for ComfyUI compatibility
            if arr.shape[2] == 1:
                # Convert grayscale to RGB
                arr = np.repeat(arr, 3, axis=2)
            elif arr.shape[2] == 4:
                # Convert RGBA to RGB (composite on white background)
                alpha = arr[:, :, 3:4]
                rgb = arr[:, :, :3]
                arr = rgb * alpha + (1 - alpha)  # White background
            
            frame_arrays.append(arr)
        
        # Stack frames: (N, H, W, C)
        stacked = np.stack(frame_arrays, axis=0)
        
        # Convert to torch tensor
        return torch.from_numpy(stacked)
