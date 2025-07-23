# Installation
1. Install pinocchio from https://stack-of-tasks.github.io/pinocchio/download.html
1. Install Visualizer

   1.1 If you use Geppeto-gui, execute the followings:
   ```
   sudo apt install robotpkg-pinocchio robotpkg-py38-qt5-gepetto-viewer-corba
   sudo apt install robotpkg-example-robot-data
   python3 -m pip install robot_descriptions
   ```
   1.1 If you use Meshcat, execute the followings:
   ```
   python3 -m pip install
   ```
1. git clone pinocchio_tools in your catkin workspace and catkin build
