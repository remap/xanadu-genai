import sys
# instructions_path = "/opt/ml/InstantID/llm_instructions/"
instructions_path = "/opt/ml/model/code/utils/llm_instructions_folder/"
sys.path.append(instructions_path)

image_description_instructions_file = "image_description_instructions.md"
keypoints_instructions_file = "keypoints_instructions.md"

with open(instructions_path+image_description_instructions_file, "r", encoding="utf-8") as f:
    image_description_instruction = f.read()

with open(instructions_path+keypoints_instructions_file, "r", encoding="utf-8") as f:
    keypoints_instruction = f.read()

# Import your examples as Python variables
from dance_01 import dance_01
from laying_05 import laying_05
from sitting_06 import sitting_06
from sitting_15 import sitting_15
from standing_04 import standing_04
from standing_05 import standing_05
from tpose_01 import tpose_01
from tpose_keypoints import tpose_keypoints
from default_pose_keypoints import DEFAULT_POSES_KEYPOINTS

# Store in a list
examples = [dance_01, laying_05, sitting_06, standing_04, standing_05, tpose_01, sitting_15, laying_05]
# Build the full prompt
keypoints_prompt = f"{keypoints_instruction}\n\nHere are some examples:\n"
for i, example in enumerate(examples, 1):  # start count at 1
    keypoints_prompt += f"\n### Example {i}:\n{example.strip()}\n"