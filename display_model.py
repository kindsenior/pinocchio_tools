from pathlib import Path
from robot_wrapper import *
from robot_descriptions.loaders.pinocchio import load_robot_description

# # UR5
robot = load_robot_description("ur5_description")
model, data = robot.model, robot.data

# # JAXON_BLUE
# urdf_path = Path("models/JAXON_BLUE.urdf")
# mesh_dir  = Path("models/JAXON_BLUE_meshes")
# package_dirs = [Path("/home/k-kojima/ros/agent_system_ws/src/pinocchio_tutorial")]

# robot = RobotWrapper.BuildFromURDF(
#     urdf_path,
#     package_dirs = package_dirs,
#     root_joint = pin.JointModelFreeFlyer()
# ) # package_dirs


# robot.setVisualizer(MeshcatVisualizer())
robot.setVisualizer(GepettoVisualizer())
robot.initViewer(loadModel=True)

# set joint angles
q = robot.q0
robot.display(q)

robot.add_fixed_frame_with_axis("force_local", "wrist_3_joint", offset_pos=(0, 0, 0.1))
robot.display_with_frames(q)
