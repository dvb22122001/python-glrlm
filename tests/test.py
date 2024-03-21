from glrlm import GLRLM
import cv2

IMG_PATH = 'assets/eiproject-logo.jpg'

"""img = cv2.imread(IMG_PATH)
app = GLRLM()
glrlm = app.get_features(img, 8)
print(glrlm.Features)"""
img = cv2.imread(IMG_PATH)
if img is not None:
    app = GLRLM()
    glrlm_features = app.get_features(img, 8)

    # If SRE is now a dictionary, you should print it accordingly
    if isinstance(glrlm_features.SRE, dict):
        print("SRE for each angle:")
        for angle, value in glrlm_features.SRE.items():
            print(f"Angle {angle}: {value}")
    else:
        print("SRE1:", glrlm_features.SRE)

    # Print other features
    print("LRE1:", glrlm_features.LRE)
    print("GLU:", glrlm_features.GLU)
    print("RLU:", glrlm_features.RLU)
    print("RPC:", glrlm_features.RPC)
else:
    print(f"Failed to load image from {IMG_PATH}")