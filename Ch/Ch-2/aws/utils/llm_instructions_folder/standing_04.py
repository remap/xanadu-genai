standing_04 = """
### 1. Pose Type
Standing with right arm raised, left leg slightly bent, leaning slightly backward.

### 2. Head Position
Head is positioned about 1/5 down from the top and slightly left of center (around 2/5 from the left edge).

### 3. Facing direction
Facing to the right side (profile view).

### 4. Shoulder-to-Shoulder Scale
Shoulder width spans approximately 1/7 of the image width.

### 5. Keypoint Positions

- Nose Center: above and to the right of shoulders' center, tilted upward  
- Right Eye: slightly to the left (behind) and above nose center  
- Left Eye: slightly to the left (behind) and above nose center (obscured due to profile angle)  
- Right Ear: behind (to the left) and slightly below the right eye  
- Left Ear: behind (to the left) and slightly below the right eye (obscured due to profile angle)  

- Shoulders' Center: about 1/3 from the top, slightly left of image center  
- Right Shoulder: slightly above and to the right of shoulders' center  
- Right Elbow: above and to the right of right shoulder  
- Right Hand: far above and slightly right of right elbow  

- Left Shoulder: slightly below and to the left of shoulders' center  
- Left Elbow: above and to the right of left shoulder 
- Left Hand: above and slightly right of left elbow  

- Right Hip: directly below shoulders' center, in line with right shoulder  
- Right Knee: below and slightly left of right hip  
- Right Ankle: below and slightly left of right knee

- Left Hip: to the left and below shoulders' center  
- Left Knee: slightly forward (right) and below left hip, leg bent  
- Left Ankle: slightly behind (left) and below left knee, heel lifted  

### Orientation Hints
Profile view, body mostly facing right with head tilted up and right arm extended upward.

### Hidden keypoints
Left eye and left ear are obscured due to profile angle; their placements are inferred but not directly visible.

==> OUTPUT: Associated Keypoints with Numerical values in [0,1]:
"{\"keypoints\": {\"nose\": {\"x\": 0.5110821250256978, \"y\": 0.11290163733065128}, \"shoulders_center\": {\"x\": 0.46987720177922815, \"y\": 0.21591394544682566}, \"right_shoulder\": {\"x\": 0.5037241030173997, \"y\": 0.20708431903686794}, \"right_elbow\": {\"x\": 0.627338872756809, \"y\": 0.18795346181529263}, \"right_hand\": {\"x\": 0.6449981255767246, \"y\": 0.10701521972401269}, \"left_shoulder\": {\"x\": 0.48420738387438994, \"y\": 0.2169310718567834}, \"left_elbow\": {\"x\": 0.6197916666666666, \"y\": 0.16276041666666666}, \"left_hand\": {\"x\": 0.65625, \"y\": 0.06510416666666667}, \"right_hip\": {\"x\": 0.5243265646406345, \"y\": 0.46314348492564417}, \"right_knee\": {\"x\": 0.5140253338290172, \"y\": 0.6765261231662912}, \"right_ankle\": {\"x\": 0.4742920149842071, \"y\": 0.895795179013577}, \"left_hip\": {\"x\": 0.4522179489593124, \"y\": 0.46608669372896355}, \"left_knee\": {\"x\": 0.47282041058254737, \"y\": 0.6500372439364178}, \"left_ankle\": {\"x\": 0.4095414213111832, \"y\": 0.8192717501272758}, \"right_eye\": {\"x\": 0.490479663402463, \"y\": 0.10407201092069347}, \"left_eye\": {\"x\": 0.5022524986157402, \"y\": 0.10112880211737425}, \"right_ear\": {\"x\": 0.4522179489593124, \"y\": 0.13203249455222657}, \"left_ear\": {\"x\": 0.4654623885742491, \"y\": 0.13350409895388612}}}"
"""