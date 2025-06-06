import math 
import cv2
import numpy as np
import json

class MyKeypoints:
  def __init__(self, x=0.0, y=0.0):
    self.x = x
    self.y = y

  def __repr__(self):
    return f"({self.x}, {self.y})"
  
  def __add__(self, other):
    if not isinstance(other, MyKeypoints):
      return NotImplemented
    return MyKeypoints(self.x + other.x, self.y + other.y)
  
  def __sub__(self, other):
    if not isinstance(other, MyKeypoints):
      return NotImplemented
    return MyKeypoints(self.x - other.x, self.y - other.y)
  
  def __mul__(self, lambda_factor):
    return MyKeypoints(self.x * lambda_factor, self.y * lambda_factor)
  
  def norm(self):
    return math.sqrt(self.x**2 + self.y**2)

def generated_dict_to_keypoints(gen_dict):
  keypoints_map = [
                  'nose', 'shoulders_center', 
                  'right_shoulder', 'right_elbow', 'right_hand', 
                  'left_shoulder', 'left_elbow', 'left_hand',
                  'right_hip', 'right_knee', 'right_ankle',
                  'left_hip', 'left_knee', 'left_ankle',
                  'right_eye', 'left_eye',
                  'right_ear', 'left_ear'
                ]
  keypoints_list = []
  for keypoint in keypoints_map:
    keypoints_list.append(
        MyKeypoints(x=gen_dict['keypoints'][keypoint]['x'],
                    y=gen_dict['keypoints'][keypoint]['y']
                    )
    )

  return keypoints_list

def extract_keypoints(keypoints_text):
    output = keypoints_text.strip()
    # If output looks like a stringified JSON (with single quotes)
    # or is surrounded by extra quotes (like '"{...}"'), clean it:
    if output.startswith("'") or output.startswith('"'):
        output = output.strip("'\"")
    
    if output[-3] != '}':
        output += '}'
    keypoints_dict = json.loads(output)

    gen_keypoints_list = generated_dict_to_keypoints(keypoints_dict)

    return gen_keypoints_list

def draw_bodypose(canvas: np.ndarray, keypoints) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape

    
    if max(W, H) < 500:
        ratio = 1.0
    elif max(W, H) >= 500 and max(W, H) < 1000:
        ratio = 2.0
    elif max(W, H) >= 1000 and max(W, H) < 2000:
        ratio = 3.0
    elif max(W, H) >= 2000 and max(W, H) < 3000:
        ratio = 4.0
    elif max(W, H) >= 3000 and max(W, H) < 4000:
        ratio = 5.0
    elif max(W, H) >= 4000 and max(W, H) < 5000:
        ratio = 6.0
    else:
        ratio = 7.0

    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            print("KEYPOINT NONE")
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), int(stickwidth * ratio)), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), int(4 * ratio), color, thickness=-1)

    return canvas
