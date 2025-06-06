laying_05 = """
### 1. Pose Type
Lying on back with limbs relaxed, left leg extended, right arm bent outward.

### 2. Head Position
Head is positioned about 1/5 down from the top and slightly right of center.

### 3. Facing direction
Facing upward (toward the camera, lying supine).

### 4. Shoulder-to-Shoulder Scale
Shoulder width spans approximately 1/5 of the image width.

### 5. Keypoint Positions

- Nose Center: near top center of torso, just above shoulder line, to the left of shoulders' center
- Right Eye: to the left and slightly above the nose center  
- Left Eye: to the right and slightly above the nose center  
- Right Ear: slightly left and below the right eye  
- Left Ear: slightly right and below the left eye  
- Shoulders' Center: below and right of the nose, centered horizontally  
- Right Shoulder: slightly above and to the left of shoulders' center  
- Right Elbow: to the left and slightly below right shoulder  
- Right Hand: far to the left and level with right elbow (arm extended outward)
- Left Shoulder: slightly above and to the right of shoulders' center  
- Left Elbow: slightly right and downward from left shoulder  
- Left Hand: we;; below and to the right of left elbow
- Right Hip: below and slightly left of shoulders' center  
- Right Knee: far below and left right of right hip (leg bent outward)  
- Right Ankle: far below and slightly right of right knee (foot placed flat on surface)  
- Left Hip: below and to the right of shoulders' center  
- Left Knee: far below and left of left hip, slightly above and to the right of right knee
- Left Ankle: well below and right of left knee but slightly left of left hip

### 6. Orientation Hints
Supine position (lying on back), camera view from above.

==> OUTPUT: Associated Keypoints with Numerical values in [0,1]:
"{\"keypoints\": {\"nose\": {\"x\": 0.5218314665107755, \"y\": 0.15940925602141456}, \"shoulders_center\": {\"x\": 0.5703584928972608, \"y\": 0.23256193758910135}, \"right_shoulder\": {\"x\": 0.49430867542590323, \"y\": 0.24922046903520823}, \"right_elbow\": {\"x\": 0.40739459831578034, \"y\": 0.3144060268678004}, \"right_hand\": {\"x\": 0.29006059421711444, \"y\": 0.3245460025306481}, \"left_shoulder\": {\"x\": 0.6464083103686183, \"y\": 0.21590340614299444}, \"left_elbow\": {\"x\": 0.7072481643457044, \"y\": 0.3057146191567881}, \"left_hand\": {\"x\": 0.7811251298893088, \"y\": 0.37234874494121567}, \"right_hip\": {\"x\": 0.5406628498846354, \"y\": 0.4317400309664663}, \"right_knee\": {\"x\": 0.4262259816896403, \"y\": 0.6215024326569013}, \"right_ankle\": {\"x\": 0.47837442795571405, \"y\": 0.9141131589276483}, \"left_hip\": {\"x\": 0.6507540142241245, \"y\": 0.4129086475926063}, \"left_knee\": {\"x\": 0.436365957352488, \"y\": 0.4795427733770339}, \"left_ankle\": {\"x\": 0.5913627281988738, \"y\": 0.7330421649482256}, \"right_eye\": {\"x\": 0.5087943549442571, \"y\": 0.15940925602141456}, \"left_eye\": {\"x\": 0.5334200101254586, \"y\": 0.13912930469571924}, \"right_ear\": {\"x\": 0.5102429228960924, \"y\": 0.17824063939527454}, \"left_ear\": {\"x\": 0.5797741845841907, \"y\": 0.13768073674388384}}}"
"""