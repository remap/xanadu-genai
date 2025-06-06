test_prompt = """
### 1. Pose Type
Lying on side with bent knees and right arm resting across the torso.

### 2. Head Position
Head is about 1/6 down from the top and 1/2 from the left edge of the image.

### 3. Facing direction
Facing slightly upward and to the left (3/4 profile view).

### 4. Shoulder-to-Shoulder Scale
Shoulder width spans approximately 1/7 of the total image width.

### 5. Keypoint Positions

- Nose Center: slightly above and right of shoulders' center
- Right Eye: to the left and slightly above nose center
- Left Eye: to the right and slightly above nose center
- Right Ear: slightly behind and left of right eye
- Left Ear: partially visible, behind and right of left eye
- Shoulders' Center: about 1/3 from the top, slightly right of image center
- Right Shoulder: slightly above and to the left of shoulders' center
- Right Elbow: below and slightly left of right shoulder
- Right Hand: resting on chest, slightly above of right elbow and right of shoulders' center
- Left Shoulder: slightly below and right of shoulders' center
- Left Elbow: below and to the right of left shoulder
- Left Hand: lying on the surface, slightly above and to the right of left elbow
- Right Hip: below and to the left of shoulders' center
- Right Knee: well below and left of right hip, bent sharply
- Right Ankle: below right knee and well left behind it, foot pointed downward
- Left Hip: slightly to the right of right hip
- Left Knee: below and to the right of left hip, bent and close to right knee
- Left Ankle: level with and well left of left knee, directly under right hip

### 6. Hidden Keypoints
All keypoints are visible or reasonably inferred from context and posture.
"""