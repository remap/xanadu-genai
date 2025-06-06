sitting_06 = """
### 1. Pose Type
Seated with legs extended, leaning back with right arm resting on chair and left hand supporting head.

### 2. Head Position
Head is positioned about 1/5 down from the top and 2/3 from the left edge of the image.

### 3. Facing Direction
Facing toward the viewer, slightly turned to their left (3/4 profile).

### 4. Shoulder-to-Shoulder Scale
Shoulder width spans approximately 1/7 of the total image width.

### 5. Keypoint Positions

- Nose Center: above and left of shoulder center, slightly tilted to the person's left
- Right Eye: slightly to the right and slightly above the nose center  
- Left Eye: to the right and slightly above the nose center  
- Right Ear: to the right and slightly below the right eye  
- Left Ear: to the right and slightly below the left eye  

- Shoulders' Center: about 1/3 down from top, right of the image center  
- Right Shoulder: slightly above and left of shoulders' center  
- Right Elbow: extended below and slightly left of right shoulder  
- Right Hand: resting on thigh, slightly below and further left than left elbow  

- Left Shoulder: to the right and slightly lower than shoulders' center  
- Left Elbow: below and slightly left of left shoulder  
- Left Hand: resting on side of head, above of right elbow  

- Right Hip: well below and left of shoulders' center  
- Right Knee: extended forward and rightward from right hip (knee is distal)  
- Right Ankle: further forward (distal) and slightly downward from right knee, resting on a table  

- Left Hip: below and left of shoulders' center  
- Left Knee: bent, pointing downward and well left of left hip  
- Left Ankle: directly below left knee, foot flat on the floor  

### 6. Orientation Hints
Torso twisted slightly toward the viewer, right leg extended across frame, left leg grounded. Weight appears supported by the chair and left arm.

==> OUTPUT: Associated Keypoints with Numerical values in [0,1]:
"{\"keypoints\": {\"nose\": {\"x\": 0.7726655388121596, \"y\": 0.24318045536832264}, \"shoulders_center\": {\"x\": 0.8061008629824586, \"y\": 0.33168572523087886}, \"right_shoulder\": {\"x\": 0.7372634308671371, \"y\": 0.31005110370892064}, \"right_elbow\": {\"x\": 0.6153228368342819, \"y\": 0.3906892384725829}, \"right_hand\": {\"x\": 0.49731581035087374, \"y\": 0.4162574275439881}, \"left_shoulder\": {\"x\": 0.8749382950977802, \"y\": 0.353320346752837}, \"left_elbow\": {\"x\": 0.8375694033780343, \"y\": 0.4496927517142872}, \"left_hand\": {\"x\": 0.8356026196033106, \"y\": 0.32185180635726135}, \"right_hip\": {\"x\": 0.5759871613398125, \"y\": 0.5342644540273964}, \"right_knee\": {\"x\": 0.3734084325432949, \"y\": 0.4162574275439881}, \"right_ankle\": {\"x\": 0.1078926229556263, \"y\": 0.465427021912075}, \"left_hip\": {\"x\": 0.6644924312023687, \"y\": 0.58736761594493}, \"left_knee\": {\"x\": 0.34980702724661333, \"y\": 0.6444043454119108}, \"left_ankle\": {\"x\": 0.3458734596971664, \"y\": 0.9000862361259622}, \"right_eye\": {\"x\": 0.7687319712627128, \"y\": 0.22351261762108784}, \"left_eye\": {\"x\": 0.7982337278835647, \"y\": 0.2274461851705348}, \"right_ear\": {\"x\": 0.7647984037132658, \"y\": 0.2274461851705348}, \"left_ear\": {\"x\": 0.8592040248999924, \"y\": 0.24318045536832264}}}"
"""