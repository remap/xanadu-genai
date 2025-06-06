import sys
import math
import numpy as np
from math import cos, sin, atan2

sys.path.append("/opt/ml/model/code/utils/")
sys.path.append("/opt/ml/InstantID/")
sys.path.append("/opt/ml/InstantID/gradio_demo")
from controlnet_util import openpose
from keypoints_utils import MyKeypoints

body_keypoints_map = [
                  'nose', 
                  'shoulders_center', 
                  'left_shoulder', 'left_elbow', 'left_hand',
                  'right_shoulder', 'right_elbow', 'right_hand', 
                  'left_hip', 'left_knee', 'left_ankle',
                  'right_hip', 'right_knee', 'right_ankle',
                  'right_eye', 'left_eye',
                  'right_ear', 'left_ear'
                ]

face_kps_map = [
  'kps_left_eye', 'kps_right_eye',
  'kps_nose',
  'kps_left_jaw', 'kps_right_jaw',
]

segments_map = [
  'notch_to_nose',
  'left_shoulder_to_notch',
  'left_elbow_to_shoulder',
  'left_hand_to_elbow',
  'right_shoulder_to_notch',
  'right_elbow_to_shoulder',
  'right_hand_to_elbow',
  'left_hip_to_notch',
  'right_hip_to_notch',
  'left_knee_to_hip',
  'right_knee_to_hip',
  'left_ankle_to_knee',
  'right_ankle_to_knee',
  'left_eye_to_nose',
  'right_eye_to_nose',
  'left_ear_to_eye',
  'right_ear_to_eye',
  'kps_left_eye_to_nose',
  'kps_right_eye_to_nose',
  'kps_left_jaw_to_nose',
  'kps_right_jaw_to_nose',
]

segment_points = {
  'notch_to_nose': {'start': 'nose', 'end': 'shoulders_center'},
  'left_shoulder_to_notch': {'start': 'shoulders_center', 'end': 'left_shoulder'},
  'left_elbow_to_shoulder': {'start': 'left_shoulder', 'end': 'left_elbow'},
  'left_hand_to_elbow': {'start': 'left_elbow', 'end': 'left_hand'},
  'right_shoulder_to_notch': {'start': 'shoulders_center', 'end': 'right_shoulder'},
  'right_elbow_to_shoulder': {'start': 'right_shoulder', 'end': 'right_elbow'},
  'right_hand_to_elbow': {'start': 'right_elbow', 'end': 'right_hand'},
  'left_hip_to_notch': {'start': 'shoulders_center', 'end': 'left_hip'},
  'right_hip_to_notch': {'start': 'shoulders_center', 'end': 'right_hip'},
  'left_knee_to_hip': {'start': 'left_hip', 'end': 'left_knee'},
  'right_knee_to_hip': {'start': 'right_hip', 'end': 'right_knee'},
  'left_ankle_to_knee': {'start': 'left_knee', 'end': 'left_ankle'},
  'right_ankle_to_knee': {'start': 'right_knee', 'end': 'right_ankle'},
  'left_eye_to_nose': {'start': 'nose', 'end': 'left_eye'},
  'right_eye_to_nose': {'start': 'nose', 'end': 'right_eye'},
  'left_ear_to_eye': {'start': 'left_eye', 'end': 'left_ear'},
  'right_ear_to_eye': {'start': 'right_eye', 'end': 'right_ear'},
  'kps_left_eye_to_nose': {'start': 'kps_nose', 'end': 'kps_left_eye'},
  'kps_right_eye_to_nose': {'start': 'kps_nose', 'end': 'kps_right_eye'},
  'kps_left_jaw_to_nose': {'start': 'kps_nose', 'end': 'kps_left_jaw'},
  'kps_right_jaw_to_nose': {'start': 'kps_nose', 'end': 'kps_right_jaw'},
}

def get_openpose_results(image):
  pose = openpose.detect_poses(np.array(image), include_hand=False)
  body_keypoints_dict = {}
  for i,keypoint in enumerate(body_keypoints_map):
    body_keypoints_dict[keypoint] = MyKeypoints(x=pose[0][0][0][i].x, y=pose[0][0][0][i].y)

  return body_keypoints_dict

def refactor_openpose_keypoints(keypoints_list):
  body_keypoints_dict = {}
  for i,keypoint in enumerate(body_keypoints_map):
    body_keypoints_dict[keypoint] = keypoints_list[i]

  return body_keypoints_dict

def refactor_face_keypoints(face_kps):
  face_kps_dict = {} 
  for i,keypoint in enumerate(face_kps_map):
    face_kps_dict[keypoint] = MyKeypoints(x=face_kps[i,0], y=face_kps[i,1])

  return face_kps_dict
  

def get_segments_scales(all_keypoints):#openpose_results, face_kps):

  # all_keypoints = {}
  # for i,keypoint in enumerate(body_keypoints_map):
  #   all_keypoints[keypoint] = MyKeypoints(x=openpose_results[0][0][0][i].x, y=openpose_results[0][0][0][i].y)

  # for i,keypoint in enumerate(face_kps_map):
  #   all_keypoints[keypoint] = MyKeypoints(x=face_kps[i,0], y=face_kps[i,1])

  segments_lengths = {}
  for i,segment in enumerate(segment_points):
    start_point = all_keypoints[segment_points[segment]['start']]
    end_point = all_keypoints[segment_points[segment]['end']]
    segments_lengths[segment] = (end_point - start_point).norm()

  normalized_segments_lengths = {}
  for i,segment in enumerate(segments_lengths):
    normalized_segments_lengths[segment] = segments_lengths[segment] / segments_lengths['notch_to_nose']

  return normalized_segments_lengths, segments_lengths



def rescale_pose_keypoints(image, face_kps, tgt_keypoints_list):

  # get openpose results for muse image
  original_body_keypoints = get_openpose_results(image)
  # print(f"ORIGINAL BODY KEYPOINTS\n {original_body_keypoints}")

  # refactor face keypoints
  original_face_keypoints = refactor_face_keypoints(face_kps)

  # refactor target keypoints
  tgt_keypoints = refactor_openpose_keypoints(tgt_keypoints_list)
  tgt_keypoints = tgt_keypoints | original_face_keypoints
  # print(f"TARGET KEYPOINTS\n {tgt_keypoints}")

  # get segments scales for original body
  normalized_segments_lengths, original_segments_lengths = get_segments_scales(original_body_keypoints | original_face_keypoints)# original_body_keypoints, original_face_keypoints)
  # print(f"ORIGINAL SEGMENTS LENGTHS\n {original_segments_lengths}")

  # get lengths for target body
  normalized_tgt_segments_lengths, tgt_segments_lengths = get_segments_scales(tgt_keypoints)
  # print(f"TARGET SEGMENTS LENGTHS\n {tgt_segments_lengths}")
  tgt_segments_lengths['notch_to_nose'] = original_segments_lengths['notch_to_nose']#/1024 # <-------- this is a hack
  # print(f"NEW TARGET SEGMENTS LENGTHS\n {tgt_segments_lengths}")
  tgt_nose_to_notch = tgt_segments_lengths['notch_to_nose']

  # rescale new keypoints
  rescaled_keypoints = {
    'nose': tgt_keypoints['nose'],
    'kps_nose': tgt_keypoints['kps_nose'],
  }
  for segment in segments_map:
    # if segment length is less than original segment length, then it's fine -- this is accounting for possible out-of-plane limb placement
    if normalized_tgt_segments_lengths[segment] <= normalized_segments_lengths[segment]:
        # print(f"{segment} is within allowed range")
        # rescaled_keypoints[segment_points[segment]['end']] = tgt_keypoints[segment_points[segment]['end']]
        rescale_factor = tgt_segments_lengths[segment] 
        # continue
    else:
        # print(f"{segment} is out of allowed range -- rescaling")
        rescale_factor = tgt_nose_to_notch * normalized_segments_lengths[segment]
        # print(f"{segment} rescale factor: {rescale_factor}")
    
    tgt_vector = tgt_keypoints[segment_points[segment]['end']] - tgt_keypoints[segment_points[segment]['start']]
    tgt_angle = atan2(tgt_vector.y, tgt_vector.x)
    rescaled_keypoints[segment_points[segment]['end']] = rescaled_keypoints[segment_points[segment]['start']] + MyKeypoints(x=cos(tgt_angle), y=sin(tgt_angle)) * rescale_factor
    # print(f"{segment} rescaled keypoints: {rescaled_keypoints[segment_points[segment]['end']]}")
  
  # return a list of body keypoints and array of face kps
  final_body_keypoints_list = [rescaled_keypoints[keypoint] for keypoint in body_keypoints_map]
  final_face_keypoints = face_kps # np.array([[rescaled_keypoints[keypoint].x, rescaled_keypoints[keypoint].y] for keypoint in face_kps_map])

  # print(f"original body keypoints: {original_body_keypoints}")
  # print(f"original face keypoints: {original_face_keypoints}")
  # print(f"final body keypoints: {rescaled_keypoints}")
  # print(f"final face keypoints: {final_face_keypoints}")

  return final_body_keypoints_list, final_face_keypoints


def rescale_face_box(face_box, original_face_kps, rescaled_face_kps):
  x1, y1, x2, y2 = face_box
  cx = (x1 + x2) / 2
  cy = (y1 + y2) / 2

  v1 = rescaled_face_kps[face_kps_map.index('kps_right_eye'), :] - rescaled_face_kps[face_kps_map.index('kps_nose'), :]
  v2 = original_face_kps[face_kps_map.index('kps_right_eye'), :] - original_face_kps[face_kps_map.index('kps_nose'), :]
  scaling_factor = np.sqrt(v1.dot(v1)) / np.sqrt(v2.dot(v2))
  print(f"scaling factor: {scaling_factor}")
  
  #  left corner
  new_x1 = cx - (x2 - x1) / 2 * scaling_factor
  new_y1 = cy - (y2 - y1) / 2 * scaling_factor

  # right corner
  new_x2 = cx + (x2 - x1) / 2 * scaling_factor
  new_y2 = cy + (y2 - y1) / 2 * scaling_factor

  return int(new_x1), int(new_y1), int(new_x2), int(new_y2)