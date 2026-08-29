"""
Region Boundary Visualization Script.
Reads 'ref.png' and the zone coordinates from 'config.py',
and generates 'ref_regions.png' with overlays, vertex dots, and coordinates.

Usage:
    python draw_regions.py
"""
import os
import sys
import cv2
import numpy as np
import importlib

# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def draw_boundaries(
    ref_image_path="ref.png",
    output_image_path="ref_regions.png",
    output_crop_path="ref_regions_crop.png"
):
    # Reload config in case it was modified in an interactive session
    importlib.reload(config)
    
    # 1. Load reference image
    if not os.path.isabs(ref_image_path):
        ref_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ref_image_path)
    if not os.path.isabs(output_image_path):
        output_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_image_path)
    if not os.path.isabs(output_crop_path):
        output_crop_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_crop_path)

    img = cv2.imread(ref_image_path)
    if img is None:
        print(f"[!] ERROR: Could not read reference image from '{ref_image_path}'")
        return

    full_h, full_w = img.shape[:2]
    print(f"[+] Loaded reference image: {full_w}x{full_h} from {ref_image_path}")

    # 2. Compute AOI Pixel Bounding Box from config.ROI_COORDS
    rx1 = int(full_w * config.ROI_COORDS["x_min_ratio"])
    rx2 = int(full_w * config.ROI_COORDS["x_max_ratio"])
    ry1 = int(full_h * config.ROI_COORDS["y_min_ratio"])
    ry2 = int(full_h * config.ROI_COORDS["y_max_ratio"])
    aoi_w = rx2 - rx1
    aoi_h = ry2 - ry1

    # 3. Create full-frame overlay
    vis_full = img.copy()
    overlay_full = img.copy()

    # Draw AOI bounding box
    cv2.rectangle(vis_full, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
    cv2.putText(vis_full, f"AOI Crop Area ({aoi_w}x{aoi_h})", (rx1 + 8, ry1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    # Zone configurations: (Name, Norm_Polygon, Color_BGR, Prefix)
    zones = [
        ("PINK Zone (Main Aisle)", config.PINK_ZONE_NORM, (255, 0, 255), "P"),
        ("RED Zone (Transition Corridor)", config.RED_ZONE_NORM, (0, 0, 255), "R"),
        ("GREEN Zone (Checkout 1 Counter)", config.GREEN_ZONE_NORM, (0, 255, 0), "G"),
    ]
    if getattr(config, "ENABLE_YELLOW_ZONE", False) and hasattr(config, "YELLOW_ZONE_NORM"):
        zones.append(("YELLOW Zone (Packing Area)", config.YELLOW_ZONE_NORM, (0, 255, 255), "Y"))

    # Helper function to convert normalized AOI points to full-image pixel coords
    def norm_to_full_px(norm_pts):
        pts_px = []
        for pt in norm_pts:
            x = int(rx1 + pt[0] * aoi_w)
            y = int(ry1 + pt[1] * aoi_h)
            pts_px.append([x, y])
        return np.array(pts_px, dtype=np.int32)

    # 4. Draw translucent polygons & vertex points
    for zone_name, norm_pts, color, prefix in zones:
        poly_px = norm_to_full_px(norm_pts)
        
        # Fill polygon on overlay
        cv2.fillPoly(overlay_full, [poly_px], color)

    # Blend translucent overlay with base image (alpha = 0.35)
    cv2.addWeighted(overlay_full, 0.35, vis_full, 0.65, 0, vis_full)

    # Draw borders, vertices, and coordinate labels
    for zone_name, norm_pts, color, prefix in zones:
        poly_px = norm_to_full_px(norm_pts)
        
        # Solid polygon outline
        cv2.polylines(vis_full, [poly_px], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Draw vertex points and labels
        for idx, (pt_norm, pt_px) in enumerate(zip(norm_pts, poly_px)):
            x, y = pt_px[0], pt_px[1]
            
            # Vertex circle (outer black, inner colored dot)
            cv2.circle(vis_full, (x, y), 6, (0, 0, 0), -1)
            cv2.circle(vis_full, (x, y), 4, color, -1)
            
            # Coordinate label: e.g. "G0 [0.125, 0.400]"
            label = f"{prefix}{idx} [{pt_norm[0]:.4f}, {pt_norm[1]:.4f}]"
            font_scale = 0.45
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            # Smart label placement offset
            lx = x + 10 if (x + tw + 15) < full_w else x - tw - 10
            ly = y - 8 if (y - th - 10) > 0 else y + th + 15
            
            cv2.rectangle(vis_full, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (15, 15, 15), -1)
            cv2.rectangle(vis_full, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), color, 1)
            cv2.putText(vis_full, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Top Header Banner & Legend
    cv2.rectangle(vis_full, (0, 0), (full_w, 48), (20, 20, 20), -1)
    cv2.line(vis_full, (0, 48), (full_w, 48), (60, 60, 60), 1)
    
    cv2.putText(vis_full, "QUEUE REGIONS CALIBRATION (Adjust values in config.py & re-run)",
                (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Legend items
    legend_items = [
        ("PINK: Arrival/Aisle", (255, 0, 255)),
        ("RED: Corridor", (0, 0, 255)),
        ("GREEN: Checkout 1", (0, 255, 0)),
    ]
    if getattr(config, "ENABLE_YELLOW_ZONE", False):
        legend_items.append(("YELLOW: Packing", (0, 255, 255)))

    lx_pos = full_w - (800 if getattr(config, "ENABLE_YELLOW_ZONE", False) else 620)
    for ltext, lcolor in legend_items:
        cv2.circle(vis_full, (lx_pos, 24), 7, lcolor, -1)
        cv2.putText(vis_full, ltext, (lx_pos + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        lx_pos += 190

    # 6. Save Full Image
    cv2.imwrite(output_image_path, vis_full)
    print(f"[+] Saved full-frame calibrated image to: {output_image_path}")

    # 7. Save AOI Zoomed Crop Image
    crop_vis = vis_full[ry1:ry2, rx1:rx2]
    if crop_vis.size > 0:
        cv2.imwrite(output_crop_path, crop_vis)
        print(f"[+] Saved AOI zoomed crop image to:      {output_crop_path}")

if __name__ == "__main__":
    draw_boundaries()
