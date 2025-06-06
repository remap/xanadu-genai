## ✅ General Instruction for Pose Description

> **Your task is to describe the pose of a single person in an image in a consistent and structured way. Your description must allow reconstruction of the body pose as a skeleton of 2D keypoints (like those from OpenPose). Include details for all visible parts, including eyes and ears if discernible. Follow this step-by-step format strictly:**

---

### 📌 Step-by-Step Format:

1. **Pose Type**  
   - Start with a short high-level label of the pose: e.g., "standing", "sitting", "jumping", "leaning", "crouched", "reaching upward", etc.  
   - Optionally include modifiers like: "arms raised", "crossed arms", "looking left", "bent knees", etc.

2. **Head Position**  
   - Describe the position of the **center of the head** in relation to the image frame (using relative terms like “1/4 down from the top and 1/3 from the left”).

3. **Facing Direction**  
   - Describe the facing direction of the person in relation to the image frame (facing forward, facing backwards, facing the right side, facing the left side).

4. **Shoulder-to-Shoulder Scale**  
   - Estimate how much of the image width the shoulders span (e.g., “shoulder width is 1/6 of the total image width”).

5. **Keypoint Relationships (use precise relative language)**  
   For each keypoint, describe its position relative to a logical parent joint, using the image reference frame (reference to body parts should still be in the person's frame, i.e. the left shoulder is the person's actual left shoulder).
   - Nose Center: relative to image top left corner
   - Shoulders' Center (middle point between the two shoulders): relative to nose center
   - Left Eye & Right Eye: relative to head center 
   - Left Ear & Right Ear: relative to eye  
   - Shoulders: relative to shoulders' center  
   - Elbows: relative to shoulders  
   - Hands: relative to elbows  
   - Hips: relative to shoulders’ center  _and_ relative to shoulders
   - Knees: relative to hips  
   - Ankles: relative to knees  
   - Examples: for a front-facing person, if the person looks to their right, in the image frame they appear to be looking to the left; if their right hand is more distal relative to their right elbow, then in the image frame the right hand keypoint is left of the right elbow's. If the person is facing backward, if their right hand is more distal relative to their right elbow, then in the image frame the right hand keypoint is right of the right elbow's.

6. **Use Consistent Spatial Terms**  
   - Use spatial references like:
     - "above", "below", "to the right of", "to the left of", "distal", "proximal"
     - Combine directions: e.g., "below and to the right"
     - Include relative distances: e.g., "slightly", "far", "directly"

7. **(Optional) Orientation Hints**  
   - Include body orientation if not frontal: “facing away”, “profile view”, “turned 3/4 to the left”, etc.

8. **Hidden keypoints**  
   - Infer keypoint placements if not visible.
---

## 🧠 Tip for Higher Quality:

- Try to use a **coordinate-free, relative language** style so the description generalizes.
- Do not give pixel coordinates or absolute positions.
- Make sure the same description pattern works across various poses.

---

## ✅ Output Format Template

```markdown
### 1. Pose Type
Standing with right arm raised, slight lean forward.

### 2. Head Position
Head is positioned about 1/4 down from the top and 3/5 from the left edge.

### 3. Facing direction
Facing forward.

### 4. Shoulder-to-Shoulder Scale
Shoulder width spans approximately 1/6 of the image width.

### 5. Keypoint Positions

- Nose Center: directly above shoulders’ center, slightly tilted forward
- Right Eye: to the left and slightly above the nose center
- Left Eye: to the right and slightly above the nose center 
- Right Ear: to the left and slightly behind right eye  
- Left Ear: to the right and slightly behind left eye  
- Shoulders’ Center: ~1/3 from the top, slightly right of image center  
- Right Shoulder: higher and to the left of shoulders’ center  
- Right Elbow: above and to the left of right shoulder  
- Right Hand: above and to the left of right elbow  
- Left Shoulder: lower and to the right of shoulders’ center  
- Left Elbow: below and slightly right of left shoulder  
- Left Hand: below and slightly left of left elbow  
- Right Hip: directly below shoulders’ center, right of right shoulder 
- Right Knee: below and slightly right of right hip  (knee more proximal than hip)
- Right Ankle: directly below right knee  
- Left Hip: to the right of shoulders’ center, left of left shoulder
- Left Knee: below and slightly right of left hip  (knee more distal than hip)
- Left Ankle: behind and right of left knee  (ankle more distal than knee)
```

---

Following these instructions, generate a description for the attached image.