# Analysis of Restraint Stress Behaviour from DeepLabCut Confidence Data

# Struggling and Freezing Scores

## Rationale & Justification

Manual scoring of "struggling" and "freezing" in a restraint stress tube is impossible to define. To overcome this, we developed a completely novel scripts that analyses the output of DeepLabCut (DLC). 

Because mice are constrained to a fixed spatial position within the tube, we discard traditional coordinate data and instead use the **confidence estimates** of tracked body parts. The camera and tube positions remain constant; therefore, any movement (e.g., rapid paw shifts) means fluctuations in confidence values. Specifically, we extract three core features from the DLC confidence data:

1. **Drop Intensity** – sudden, sharp decreases in confidence.
2. **Volatility** – rapid, high-frequency fluctuations over a short time window.
3. **Low Confidence** – sustained periods of reduced detection reliability.

These features are are mechanistically linked to the animal's 'fight or flight' stress response.

---

## Unbiased Weighting & Sensitivity

To prevent arbitrary thresholding from skewing results, our algorithms test multiple sensitivities for each video. Instead of imposing a universal rule, the code weights each of the three features according to **how much that feature actually explains the raw confidence data for that specific mouse**. 

- For **Struggling**, the weights are dynamically adjusted per mouse to determine which feature (e.g., drop intensity vs. volatility) best captures its unique movement profile.
- For **Freezing**, the algorithm tests multiple sensitivity values per video and applies a relative normalization of confidence scores to objectively measure stability.

This ensures our metrics are driven by the data itself, rather than human preconceptions of what constitutes a stress behaviour.

All analyses are performed using custom Python scripts. The data are scaled, filtered, and subjected to minimum bout thresholds to remove noise.
