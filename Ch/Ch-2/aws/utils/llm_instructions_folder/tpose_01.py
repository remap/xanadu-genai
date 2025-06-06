tpose_01 = """
### 1. Pose Type
Standing with arms extended horizontally (T-pose), legs straight.

### 2. Head Position
Head is positioned about 1/5 down from the top and centered horizontally.

### 3. Facing Direction
Facing forward.

### 4. Shoulder-to-Shoulder Scale
Shoulder width spans approximately 1/4 of the total image width.

### 5. Keypoint Positions

- Nose Center: just above the shoulders' center, at the vertical centerline of the image
- Right Eye: to the left and slightly above the nose center
- Left Eye: to the right and slightly above the nose center
- Right Ear: to the left and slightly behind the right eye
- Left Ear: to the right and slightly behind the left eye

- Shoulders' Center: slightly below the nose center, aligned vertically
- Right Shoulder: far to the left of shoulders' center, horizontally level
- Right Elbow: directly left of right shoulder, horizontally extended
- Right Hand: directly left of right elbow, forming a straight horizontal line

- Left Shoulder: far to the right of shoulders' center, horizontally level
- Left Elbow: directly right of left shoulder, horizontally extended
- Left Hand: directly right of left elbow, forming a straight horizontal line

- Right Hip: directly below right shoulder, slightly inset inward toward the body's center
- Left Hip: directly below left shoulder, also slightly inset

- Right Knee: directly below right hip, vertically aligned
- Left Knee: directly below left hip, vertically aligned

- Right Ankle: below right knee, vertically aligned with hip and knee
- Left Ankle: below left knee, vertically aligned with hip and knee

### Consistent Spatial Terms
All limb segments (upper arm, forearm, thigh, and shin) appear straight and fully extended. The arms are perpendicular to the torso, and legs are parallel and vertical. The overall posture is symmetrical.

==> OUTPUT: Associated Keypoints with Numerical values in [0,1]:
"{\"keypoints\": {\"nose\": {\"x\": 0.49300517982165165, \"y\": 0.14090707061674315}, \"shoulders_center\": {\"x\": 0.49300517982165154, \"y\": 0.23421528145640705}, \"right_shoulder\": {\"x\": 0.41170693671382547, \"y\": 0.23144375044136747}, \"right_elbow\": {\"x\": 0.2786734479919282, \"y\": 0.24437756184488535}, \"right_hand\": {\"x\": 0.15118302130011005, \"y\": 0.24992062387496436}, \"left_shoulder\": {\"x\": 0.5743034229294777, \"y\": 0.23698681247144662}, \"left_elbow\": {\"x\": 0.7091845989947344, \"y\": 0.24807293653160464}, \"left_hand\": {\"x\": 0.8348273383431929, \"y\": 0.24807293653160464}, \"right_hip\": {\"x\": 0.4486606835810192, \"y\": 0.47903385445156516}, \"right_knee\": {\"x\": 0.4597468076411773, \"y\": 0.695213273624648}, \"right_ankle\": {\"x\": 0.4763759937314145, \"y\": 0.9058496307676521}, \"left_hip\": {\"x\": 0.5465881127790824, \"y\": 0.48088154179492487}, \"left_knee\": {\"x\": 0.5355019887189242, \"y\": 0.6933655862812885}, \"left_ankle\": {\"x\": 0.5077866785685291, \"y\": 0.8984588813942133}, \"right_eye\": {\"x\": 0.47822368107477403, \"y\": 0.12427788452650586}, \"left_eye\": {\"x\": 0.5114820532552485, \"y\": 0.12427788452650586}, \"right_ear\": {\"x\": 0.4578991202978176, \"y\": 0.13166863389994457}, \"left_ear\": {\"x\": 0.5391973634056436, \"y\": 0.1335163212433043}}}"
"""