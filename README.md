# MAIXCAM-Drone-detector
This repository contains code for real-time object tracking and autonomous aerial guidance deployed on drone-mounted Sipeed MaixCAM edge hardware. Leveraging optimized YOLO26 models, the pipeline detects and tracks specified targets, calculates spatial error vectors, and outputs active flight corrections via serial protocol to align the UAV with the target object.  
Structured (Key Features)
-->Edge AI Tracking: Deploys YOLO26 object detection and tracking directly on the Sipeed MaixCAM vision board mounted on an aerial vehicle.
-->Target Guidance Logic: Continuously calculates positional errors and flight adjustments relative to the tracked target.
-->Serial Telemetry Output: Transmits control correction signals via UART/Serial interface directly to the flight controller for active target intercept and centering.
