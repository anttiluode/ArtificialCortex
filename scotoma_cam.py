import cv2
import numpy as np
import time

def run_scotoma_cam():
    print("Booting Video Tensor: Scintillating Scotoma (Broken Gate) Simulation...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Read a frame to get dimensions
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    
    # Pre-compute the spatial latent grid for the webcam dimensions
    x = np.linspace(-10, 10, w)
    y = np.linspace(-10 * (h/w), 10 * (h/w), h)
    X, Y = np.meshgrid(x, y)
    
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    
    # Grid/Lattice frequency (The V1 Cortical Columns)
    k_hex = 8.0 
    v1_lattice = np.cos(k_hex * X) * np.cos(k_hex * Y)
    
    start_time = time.time()
    aura_radius = 0.0
    aura_expansion_rate = 0.5 # How fast the cortical spreading depression moves
    
    print("Running. Press 'q' to quit, 'r' to reset the spreading depression.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Mirror for natural feel
        t = time.time() - start_time
        
        # 1. The Raw Topological Engine (Chiral Spiral)
        # Fast rotating, tightly wound phase field
        omega = 15.0
        k_spiral = 4.0
        spiral = np.cos(omega * t - k_spiral * R + 3 * Theta)
        
        # 2. The Moiré Scintillation (Engine beating against V1 Lattice)
        # This creates the jagged "fortification" zig-zag effect
        scintillation = np.sin(10 * (spiral + v1_lattice))
        # Normalize to 0-1
        scintillation = (scintillation + 1.0) / 2.0
        
        # Expand the Cortical Spreading Depression (Aura Radius)
        aura_radius += aura_expansion_rate * 0.03
        if aura_radius > 15.0:
            aura_radius = 15.0
            
        # 3. The Broken Gate (The Wave Collapse)
        # Inside the aura radius, the inhibitory gate fails.
        # We create a smooth ring (the active edge of the scotoma)
        aura_edge = np.exp(-((R - aura_radius)**2) / 2.0)
        
        # Behind the active edge, the cortex is exhausted (blind spot / suppression)
        exhaustion = np.clip(1.0 - (aura_radius - R)/5.0, 0.2, 1.0)
        exhaustion[R > aura_radius] = 1.0 # Normal outside
        
        # 4. Collide Engine with Reality
        # Convert frame to float for math
        frame_float = frame.astype(np.float32) / 255.0
        
        # Apply the visual interference
        # Where the aura is active, multiply reality by the harsh Moiré scintillation
        scintillation_3d = np.dstack([scintillation]*3)
        aura_edge_3d = np.dstack([aura_edge]*3)
        exhaustion_3d = np.dstack([exhaustion]*3)
        
        # Blend: Reality + Scintillation at the edge * Exhaustion in the wake
        broken_reality = frame_float * (1.0 - aura_edge_3d * 0.8) + (scintillation_3d * aura_edge_3d)
        broken_reality = broken_reality * exhaustion_3d
        
        # Convert back to display format
        output_frame = np.clip(broken_reality * 255.0, 0, 255).astype(np.uint8)
        
        cv2.imshow("Video Tensor: Scintillating Scotoma", output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            aura_radius = 0.0 # Reset the depression wave

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_scotoma_cam()