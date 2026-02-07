"""
vision_nav_prototype.py

Prototype pipeline:
- Capture a frame (camera or file)
- Estimate relative depth with MiDaS
- Create an occupancy grid from depth
- Compute the "clearest" route: find largest reachable free-space frontier and A* to it
- Produce direction (angle + short natural sentence)
- Optionally send the frame+direction to a local Moondream/LLM for a richer textual description

Notes:
- This is a prototype. Depth from MiDaS is *relative* depth (not metric) but is enough to find obstacles vs free space.
- Tweak thresholds and grid resolution to match your camera & environment.
"""

import cv2
import numpy as np
import torch
import math
from scipy import ndimage
from skimage.morphology import remove_small_objects
from heapq import heappush, heappop
from PIL import Image
import Alfred_config

# ----------------------
# Depth estimation (MiDaS)
# ----------------------
def load_midas(device='cuda' if torch.cuda.is_available() else 'cpu'):
    model_type = "DPT_Large"  # or "MiDaS_small" for speed
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.to(device)
    model.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform if model_type.startswith("DPT") else torch.hub.load("intel-isl/MiDaS", "transforms").default_transform
    return model, transform, device

def estimate_depth(frame_bgr, model, transform, device):
    # frame_bgr: OpenCV BGR image
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_img = Image.fromarray(img)
    input_tensor = transform(input_img).to(device)
    with torch.no_grad():
        prediction = model(input_tensor.unsqueeze(0))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    # normalize to 0..1 for convenience
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth

# ----------------------
# Occupancy grid from depth
# ----------------------
def depth_to_occupancy(depth_map, obstacle_thresh=0.45, downscale=4, min_free_region=200):
    """
    depth_map: normalized 0..1 (closer might be smaller or larger depending on model; MiDaS maps larger -> farther)
    obstacle_thresh: threshold in normalized depth indicating obstacle (tweak for your camera/scene)
    downscale: reduce resolution for planning speed
    """
    # For MiDaS, higher values are farther. We want obstacles to be *near* the camera -> lower depth values are near.
    # So invert if necessary. We'll treat "close" as obstacle. Let's compute a closeness map:
    closeness = 1.0 - depth_map  # closeness: 1 means nearest
    # binary occupancy: True = obstacle
    occ = closeness > obstacle_thresh
    # morphological cleanup
    occ = ndimage.binary_closing(occ, structure=np.ones((5,5))).astype(np.bool_)
    # downscale
    if downscale > 1:
        small = cv2.resize(occ.astype(np.uint8)*255, (occ.shape[1]//downscale, occ.shape[0]//downscale), interpolation=cv2.INTER_NEAREST)
        occ = (small > 128)
    # remove tiny obstacles
    occ = remove_small_objects(occ, min_size=20)
    # occupancy: 1 obstacle, 0 free
    occupancy = occ.astype(np.uint8)
    return occupancy

# ----------------------
# Path / "clearest route" determination
# ----------------------
def find_largest_frontier_and_goal(occupancy):
    """
    occupancy: 2D binary grid (1 obstacle, 0 free)
    starting point is assumed center of grid
    We compute distance transform from obstacles and find the free cell with the largest distance
    that's reachable from the start (so we don't pick unreachable corner behind obstacles).
    """
    h, w = occupancy.shape
    # compute distance transform of free space to obstacles
    free = (occupancy == 0)
    dist = ndimage.distance_transform_edt(free)
    # start cell
    start = (h//2, w//2)
    if not free[start]:
        # if start is inside an obstacle (rare), try to find nearest free neighbor
        yx = np.argwhere(free)
        if len(yx) == 0:
            return None, None
        # pick nearest free pixel to center
        dists = np.sum((yx - np.array(start))**2, axis=1)
        nearest = yx[np.argmin(dists)]
        start = tuple(nearest)
    # BFS to get reachable mask
    visited = np.zeros_like(occupancy, dtype=np.bool_)
    stack = [start]
    visited[start] = True
    neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    while stack:
        y,x = stack.pop()
        for dy,dx in neighbors:
            ny,nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny,nx] and free[ny,nx]:
                visited[ny,nx] = True
                stack.append((ny,nx))
    # among visited free cells, find the one with highest dist (furthest from obstacles)
    reachable_idxs = np.argwhere(visited & free)
    if reachable_idxs.size == 0:
        return start, start
    best_idx = reachable_idxs[np.argmax(dist[visited & free])]
    goal = tuple(best_idx)
    return start, goal

# A* on grid
def astar_grid(occupancy, start, goal):
    """
    occupancy: 1 obstacle, 0 free
    start, goal: (y,x) tuples
    returns path as list of (y,x) from start to goal or None
    """
    h,w = occupancy.shape
    def h_score(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = []
    heappush(open_set, (0 + h_score(start,goal), 0, start, None))
    came_from = {}
    g_score = {start: 0}
    visited = set()
    while open_set:
        f, g, current, parent = heappop(open_set)
        if current in visited:
            continue
        came_from[current] = parent
        visited.add(current)
        if current == goal:
            # reconstruct
            path = []
            cur = current
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path
        y,x = current
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]:
            ny,nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and occupancy[ny,nx] == 0:
                tentative_g = g + math.hypot(dy,dx)
                neigh = (ny,nx)
                if tentative_g < g_score.get(neigh, 1e9):
                    g_score[neigh] = tentative_g
                    heappush(open_set, (tentative_g + h_score(neigh, goal), tentative_g, neigh, current))
    return None

# ----------------------
# Direction / natural language output
# ----------------------
def path_to_direction(path, occupancy_shape, downscale=4):
    """
    path: list of (y,x) in downscaled grid
    returns: bearing angle in degrees (0 = forward/up), descriptive text
    """
    if not path or len(path) < 2:
        return None, "No path found."
    start = np.array(path[0], dtype=float)
    goal = np.array(path[-1], dtype=float)
    vec = goal - start  # y,x
    # convert to image coordinates: y positive down; we'll define 0 deg as forward/up (-y)
    angle_rad = math.atan2(-vec[0], vec[1])  # -y (forward) vs x (right)
    angle_deg = (math.degrees(angle_rad) + 360) % 360
    # map angle to human direction
    # define forward = 0..45 or 315..360 => "forward"
    def angle_to_words(a):
        if a <= 45 or a >= 315:
            return "forward"
        if 45 < a <= 135:
            return "right"
        if 135 < a <= 225:
            return "backward"
        return "left"
    word = angle_to_words(angle_deg)
    distance_cells = np.linalg.norm(goal - start)
    # approximate metric distance if camera FOV and downscale known; otherwise just say "approximately X grid units"
    descr = f"Head {word} (bearing {angle_deg:.0f}°). Approx. {distance_cells*downscale:.0f} pixel-units to the clearest open area."
    return angle_deg, descr

# ----------------------
# Optional: send prompt + frame to Moondream / local LLM (placeholder)
# ----------------------
def send_to_moondream(frame_bgr, direction_text, llm_endpoint=None):
    """
    Placeholder for integration with Moondream or a local LLM.
    You can implement as:
      - Save frame to file and send path + text to model
      - Use an API to send base64 image + prompt
    For now, return a synthetic description.
    """
    # Example prompt you could send:
    prompt = f"The camera sees a room. Best route: {direction_text}\nPlease describe simply how to move (2-3 short steps)."
    # If you have a local LLM API, call it here. This function returns an LLM-generated sentence.
    # Example stub:
    return "Go straight for a few steps, veer slightly to the right and proceed to the open space."

# ----------------------
# Main demo flow
# ----------------------
def demo_from_image(image_path=None, use_camera=False):
    # load frame
    if use_camera:
        cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
##        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Can't read from camera")
    else:
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Can't read image {image_path}")

    # prepare MiDaS
    model, transform, device = load_midas()
    depth = estimate_depth(frame, model, transform, device)

    # compute occupancy
    OCC_DOWNSCALE = 4
    occupancy = depth_to_occupancy(depth, obstacle_thresh=0.45, downscale=OCC_DOWNSCALE)
    # find start and goal
    start, goal = find_largest_frontier_and_goal(occupancy)
    if start is None or goal is None:
        print("No valid free area detected.")
        return

    path = astar_grid(occupancy, start, goal)
    angle_deg, descr = path_to_direction(path, occupancy.shape, downscale=OCC_DOWNSCALE)

    print("Direction:", descr)

    # Optionally ask Moondream / LLM for a human-friendly description
    llm_reply = send_to_moondream(frame, descr, llm_endpoint=None)
    print("LLM suggestion:", llm_reply)

    # Visualize outcome (for debugging)
    vis = cv2.resize((1.0 - depth) * 255.0, (frame.shape[1], frame.shape[0])).astype(np.uint8)  # closeness visualization
    vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    # overlay path on a small canvas
    occ_vis = cv2.resize((occupancy * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    occ_color = cv2.cvtColor(occ_vis, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(frame, 0.6, occ_color, 0.4, 0)
    # draw start & goal on overlay scaled up
    def upscale(pt):
        y,x = pt
        return int(x * OCC_DOWNSCALE), int(y * OCC_DOWNSCALE)
    if path:
        for p in path:
            x,y = upscale(p)
            cv2.circle(overlay, (x,y), 2, (0,255,0), -1)
    sx,sy = upscale(start)
    gx,gy = upscale(goal)
    cv2.circle(overlay, (sx,sy), 6, (255,0,0), -1)  # start (blue)
    cv2.circle(overlay, (gx,gy), 6, (0,0,255), -1)  # goal (red)
    cv2.putText(overlay, f"{descr}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    # python vision_nav_prototype.py path/to/room.jpg
    import sys
    if len(sys.argv) > 1:
        demo_from_image(image_path=sys.argv[1], use_camera=False)
    else:
        print("No image path provided, capturing from camera (0).")
        demo_from_image(use_camera=True)
