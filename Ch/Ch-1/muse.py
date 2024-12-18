'''
    Second step: Use Control Net to extract muse poses/edges to later add to background image

'''

from controlnet_aux import OpenposeDetector
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np

muse_image = 'muse-images/terpiscore.png'

# Extract pose

def load_image(filepath):
    return Image.open(filepath).convert("RGB")

def make_image_grid(images, rows, cols):
    """
    Display a grid of images with matplotlib.

    Parameters
    ----------
    images : List[PIL.Image]
        List of images to display in the grid.
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    for img, ax in zip(images, axes):
        ax.imshow(np.array(img))
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def get_pose(muse_image):
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    # Extract the pose of an image
    im = load_image(muse_image)
    pose_im = openpose(im)
    # pose_im.resize((pose_im.width//2, pose_im.height//2))
    ## shows comparison of image with pose
    # make_image_grid([im.resize((pose_im.width//2, pose_im.height//2)),
    #              pose_im.resize((pose_im.width//2, pose_im.height//2))], 1,2)

    return pose_im


pose_im = get_pose(muse_image)
pose_image_path = 'muse-images/terpiscore_pose.png'
pose_im.save(pose_image_path)
print(f"Pose map saved to: {pose_image_path}")