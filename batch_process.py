"""
Batch Processing Script for Queue Counting & Wait-Time Analytics.
Processes all video files in the 'inputs/' folder and saves annotated videos to 'output/'.
"""
import os
import sys
import time
import glob
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import run_pipeline

def run_batch(
    input_dir="/Users/admin/Downloads/intel_testing/footfall_analytics/queue_counting/inputs",
    output_dir="/Users/admin/Downloads/intel_testing/footfall_analytics/queue_counting/output",
    stride=2,
    out_width=960
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported video extensions
    extensions = ["*.mkv", "*.mp4", "*.avi", "*.mov"]
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    video_files = sorted(video_files)
    
    if not video_files:
        print(f"[!] No video files found in '{input_dir}'")
        return
        
    total_videos = len(video_files)
    print("==================================================================")
    print("           SPS-QUEUE BATCH VIDEO PROCESSING PIPELINE              ")
    print("==================================================================")
    print(f"[*] Input Directory:   {input_dir}")
    print(f"[*] Output Directory:  {output_dir}")
    print(f"[*] Total Videos:      {total_videos}")
    print(f"[*] Stride:            {stride}")
    print(f"[*] Output Width:      {out_width}px (Crisp 720p)")
    print("==================================================================\n")
    
    batch_start = time.time()
    
    for idx, video_path in enumerate(video_files, 1):
        filename = Path(video_path).stem
        ext = ".mp4"  # Standardize output format to mp4
        out_path = os.path.join(output_dir, f"{filename}_annotated{ext}")
        
        print(f"\n[{idx}/{total_videos}] >>> Processing: {os.path.basename(video_path)} -> {os.path.basename(out_path)}")
        t0 = time.time()
        
        try:
            run_pipeline(
                source=video_path,
                save_video=out_path,
                display=False,
                stride=stride,
                out_width=out_width
            )
            elapsed = time.time() - t0
            print(f"[✓] Completed {os.path.basename(video_path)} in {elapsed:.1f}s")
        except Exception as e:
            print(f"[!] ERROR processing {video_path}: {e}")
            
    total_batch_time = time.time() - batch_start
    print("\n==================================================================")
    print("                 BATCH PROCESSING COMPLETED                       ")
    print("==================================================================")
    print(f"[*] Total Videos Processed: {total_videos}")
    print(f"[*] Total Elapsed Time:     {total_batch_time:.2f} seconds ({total_batch_time/60.0:.2f} mins)")
    print(f"[*] Output Folder:          {output_dir}")
    print("==================================================================")

if __name__ == "__main__":
    run_batch()
