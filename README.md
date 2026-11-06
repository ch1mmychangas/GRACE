# GRACE: Gesture-Controlled C-Arm

> *Built by the BMES@UCF GRACE EE/CS/ME teams.*

**GRACE** is a gesture-controlled mobile C-arm developed by the BMES@UCF project team. It lets a scrubbed physician position X-ray imaging equipment with hand gestures alone, with no levers, buttons, or hand-signal relay to a non-sterile operator.

## Motivation

Traditional C-arm operation depends on manual controls. Physicians must break scrub, shift focus from the surgical field, or rely on a separate operator to interpret hand signals and reposition the machine. This disrupts workflow, increases procedure time, and raises the risk of contamination. GRACE aims to give the physician sterile, contactless, direct control of the system.

## How It Works

A camera tracks the physician's hand, and a computer vision model classifies the gesture. Recognized gestures are translated into commands for a tabletop-model C-arm with five independently motorized axes:

| Axis | Motion |
|------|--------|
| Vertical | Up / Down |
| Lateral | Left / Right |
| Longitudinal | Push / Pull |
| Axial rotation | Rotation about the arm's axis |
| Orbital swing | Motion along the C-arm's arc |

### Safety-Focused Control Logic

- Motion begins only after a deliberate **start gesture**
- Only **one direction** of movement is permitted at a time
- A designated **closing gesture** stops the system

## Repository Contents

This repository contains the software and firmware for the project:

### 1. Computer Vision (Python)
Scripts for setting up, training, and integrating the gesture-recognition pipeline:
- A 14-class hand-gesture recognition model (trained on a dataset of 7,000+ images; ~95% accuracy)
- Real-time hand tracking and landmark detection with OpenCV and MediaPipe
- Mapping of recognized gestures to C-arm movement commands
- Gesture confirmation, movement, and start/stop logic
- Integration with the hardware controller

### 2. Firmware (Arduino / ESP32)
Firmware that connects the sensing and actuation hardware:
- Stepper motor and motor driver control across the five axes
- Sensor integration: limit switches, potentiometers, and time-of-flight (ToF) distance sensors
- Camera module / Seeed Grove AI board integration
- Communication between the microcontrollers, sensors, and motor drivers

## Getting Started

You can clone the repository:
```bash 
git clone https://github.com/ch1mmychangas/GRACE.git
# Enter the project directory
cd GRACE
```

To work on the vision model with Python, open the `vision/` folder in your IDE/code editor. See how to run and test the model in the `README.md` in [`vision/`](./vision)

To work on the arduino system with C++, open the `firmware/` folder in your IDE/code editor.

Opening the repository at the root will cause a bunch of errors.

```bash
# Model training
# Flashing the firmware
```

## License

TBD