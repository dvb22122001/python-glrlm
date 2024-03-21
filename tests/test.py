from glrlm import GLRLM
import cv2

IMG_PATH = 'assets/eiproject-logo.jpg'

img = cv2.imread(IMG_PATH)
if img is not None:
    app = GLRLM()
    glrlm_features = app.get_features(img, 8)

    # Print SRE for each angle
    print("SRE for each angle:")
    for angle, value in glrlm_features.SRE.items():
        print(f"Angle {angle}: {value}")

    # Similarly, print other features for each angle
    print("\nLRE for each angle:")
    for angle, value in glrlm_features.LRE.items():
        print(f"Angle {angle}: {value}")

    print("\nGLU for each angle:")
    for angle, value in glrlm_features.GLU.items():
        print(f"Angle {angle}: {value}")

    print("\nRLU for each angle:")
    for angle, value in glrlm_features.RLU.items():
        print(f"Angle {angle}: {value}")

    print("\nRPC for each angle:")
    for angle, value in glrlm_features.RPC.items():
        print(f"Angle {angle}: {value}")

else:
    print(f"Failed to load image from {IMG_PATH}")
