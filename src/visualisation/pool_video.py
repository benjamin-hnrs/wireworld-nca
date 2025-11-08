import imageio
import numpy as np
from pathlib import Path
import re
from PIL import Image



def generate_pool_video(
    input_dir: Path = Path('train_log'),
    output_path: Path = Path('training_progression.mp4'),
    fps: int = 10,
    scale: float = 1.0,
    pattern: str = '*_pool.png',
    reverse: bool = False,
    loop_back: bool = False
):
    """
    Create a video from pool images generated during training.
    
    Args:
        input_dir: Directory containing the pool images
        output_path: Output video file path
        fps: Frames per second for the output video
        scale: Scale factor for images (e.g., 0.5 for half size, 2.0 for double)
        pattern: Glob pattern for finding pool images (default: '*_pool.jpg')
        reverse: If True, play the video in reverse order
        loop_back: If True, add reversed frames to create a loop effect
    
    Returns:
        Path to the generated video file
    """
    input_path = Path(input_dir)
    
    # Find all pool images matching the pattern
    pool_files = list(input_path.glob(pattern))
    
    if not pool_files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return None
    
    # Sort files numerically based on the number in filename
    def extract_number(filepath: Path):
        # Extract number from filename like "0001_pool.jpg"
        match = re.search(r'(\d+)_pool', filepath.name)
        return int(match.group(1)) if match else 0
    
    pool_files.sort(key=extract_number)
    
    if reverse:
        pool_files = pool_files[::-1]
    
    print(f"Found {len(pool_files)} pool images")
    print(f"First: {pool_files[0].name}")
    print(f"Last: {pool_files[-1].name}")
    
    # Prepare video writer
    writer = imageio.get_writer(output_path, fps=fps)
    # Ensure output directory exists
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)    
    # Process each image
    frames_written = 0
    for i, img_path in enumerate(pool_files):
        # Load image
        img = Image.open(img_path)
        
        # Scale if needed
        if scale != 1.0:
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        frame = np.array(img)
        
        # Write frame
        writer.append_data(frame)
        frames_written += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(pool_files)} images")
    
    # Add reversed frames for loop effect
    if loop_back:
        print("Adding reversed frames for loop effect...")
        for i, img_path in enumerate(reversed(pool_files[1:-1])):  # Skip first and last to avoid duplication
            img = Image.open(img_path)
            
            if scale != 1.0:
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            frame = np.array(img)
            writer.append_data(frame)
            frames_written += 1
    
    writer.close()
    
    duration = frames_written / fps
    print(f"\nVideo created successfully!")
    print(f"  Output: {output_path}")
    print(f"  Frames: {frames_written}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  FPS: {fps}")
    print(f"  Scale: {scale}x")
    
    return Path(output_path)