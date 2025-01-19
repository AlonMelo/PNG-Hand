# Print-N-Grip: A Disposable, Compliant, Scalable, and One-Shot 3D-Printed Multi-Fingered Robotic Hand

## Overview
Print-N-Grip (PNG) is a one-shot 3D-printed, tendon-based, underactuated robotic hand that is designed to be disposable, compliant, and scalable. It can adapt to the shape of grasped objects and accommodate a variable number of fingers as required. The simple design allows easy detachment and reassembly, making it an ideal solution for hazardous environments where contamination is a concern.

## Project Highlights
- **Published Work:** This project was published in the [Journal of Mechanical Design](https://arxiv.org/pdf/2401.16463).
- **Patent:** The project is patented and more details can be found at [Ramot's website](https://ramot.org/technologies/print-n-grip-a-disposable-compliant-scalable-and-one-shot-3d-printed-multi-fingered-robotic-hand/).
- **Video Demonstration:** [Watch the Print-N-Grip Hand in Action](https://www.youtube.com/watch?v=dk5-teuzLGE)

## Features and Capabilities
- **Scalable and Low-Cost Solution:** The PNG hand provides a cost-effective and scalable solution for disposable robotic hands.
- **Validated Performance:** Experimental results have confirmed the effectiveness of the developed model.
- **Excellent Grasping Ability:** The design achieves impressive grasping results with configurations of 2, 3, and 4 fingers.
- **Wide Scalability:** The design allows for scaling up to 350% and beyond.
- **Easy Operation and Integration:** Custom software enables seamless operation and integration.
- **Enhanced Precision:** A neural network-assisted control improves torque estimation and precision.
- **Versatile and Robust Design:** The PNG hand sets a new standard for one-shot 3D-printed grippers.

## Documentation
More information, including model development, testing results, and additional details, can be found in the provided [presentation](PRINT-N-GRIP_presentation.pdf).

## Installation and Usage
To get started with the Print-N-Grip hand, follow these steps:
1. Clone the [Yale OpenHand Project](https://github.com/grablab/openhand_node) following their official instructions.
2. Clone the Print-N-Grip repository:
   ```bash
   git clone https://github.com/yourusername/Print-N-Grip.git
   ```
3. Install dependencies:
   ```bash
   sudo apt-get install ros-kinetic-desktop-full
   pip install -r requirements.txt
   ```
4. Run the example script:
   ```bash
   python main.py
   ```

## Code Base
The majority of the code used in this project is derived from the [Yale OpenHand Project](https://github.com/grablab/openhand_node), with modifications that include:
- Creation of a class tailored for the PNG hand.
- Communication adjustments for Dynamixel XH540.
- Main user files for a wide range of operations.
- Torque converter neural network for enhanced precision.

## Contributions
If you would like to contribute to the project, feel free to fork the repository and submit a pull request with your improvements.

## License
This project is released under the MIT License.

## Contact
For further inquiries, please contact [Alon Laron](mailto:your.email@example.com).

