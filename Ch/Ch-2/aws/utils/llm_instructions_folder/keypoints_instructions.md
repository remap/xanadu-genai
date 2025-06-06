### LLM Instruction for Keypoint Generation

You are given a structured human pose description in natural language. Your task is to **analyze the spatial relationships** described, then generate the **corresponding 2D keypoint coordinates** normalized to the `[0, 1]` image coordinate system.

#### Output Format
At the end of your response, **return a JSON object** in the following structure:
```json
{
  "keypoints": {
    "nose": {"x": ..., "y": ...},
    "right_eye": {"x": ..., "y": ...},
    "left_eye": {"x": ..., "y": ...},
    "right_ear": {"x": ..., "y": ...},
    "left_ear": {"x": ..., "y": ...},
    "shoulders_center": {"x": ..., "y": ...},
    "right_shoulder": {"x": ..., "y": ...},
    "right_elbow": {"x": ..., "y": ...},
    "right_hand": {"x": ..., "y": ...},
    "left_shoulder": {"x": ..., "y": ...},
    "left_elbow": {"x": ..., "y": ...},
    "left_hand": {"x": ..., "y": ...},
    "right_hip": {"x": ..., "y": ...},
    "right_knee": {"x": ..., "y": ...},
    "right_ankle": {"x": ..., "y": ...},
    "left_hip": {"x": ..., "y": ...},
    "left_knee": {"x": ..., "y": ...},
    "left_ankle": {"x": ..., "y": ...}
  }
}
```

All coordinates (`x`, `y`) must be floating-point numbers between 0 and 1, relative to the total image size:
- `x = 0` is the **left edge**, `x = 1` is the **right edge**
- `y = 0` is the **top**, `y = 1` is the **bottom**

#### Interpretation Guidelines
When estimating keypoint coordinates:
- Use **proportional reasoning** based on phrases like "center," "just above," or "slightly inset."
- Assume the subject is centered in the image unless otherwise stated.
- Use relative placement logic to infer distances and angles between joints (e.g., "directly below," "horizontally level," "symmetrical").

#### Example Structure
You will typically receive input formatted like this:
```
### 1. Pose Type
...

### 2. Head Position
...

### 3. Facing Direction
...

### 4. Shoulder-to-Shoulder Scale
...

### 5. Keypoint Positions
- Nose Center: ...
- Right Eye: ...
...

### Consistent Spatial Terms
...
```

After parsing this information, compute the normalized keypoints and produce output as follows:
```
==> OUTPUT:
"{\"keypoints\": { ... }}"
```

Make sure your output is a **single valid JSON string** suitable for parsing.
---
