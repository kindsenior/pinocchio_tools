#!/usr/bin/env -S python3 -i
import numpy as np
import pinocchio as pin
from pinocchio_tools.robot_wrapper import *

model = pin.buildSampleModelManipulator()
visual_model  = pin.buildSampleGeometryModelManipulator(model)
collision_model = visual_model.copy()
robot = RobotWrapper(model, collision_model, visual_model)
robot.setVisualizer(GepettoVisualizer())
robot.initViewer(loadModel=True)
q = robot.q0.copy()
robot.display(q)

# target frame
des_frame = pin.SE3(np.eye(3), np.array([1.0, 0.0, 1.0]))

# add XYZaxis
node_dict = {}
axis_radius = 0.05; axis_size = 0.2
node_name = "world/target"
robot.viewer.gui.addXYZaxis(node_name, [1., 0., 0., 1.], axis_radius, axis_size)
node_dict[node_name] = des_frame

# q = robot.inverse_kinematics("wrist2_joint", des_frame)
q = robot.inverse_kinematics("wrist2_joint", des_frame, robot.get_joint_names('shoulder1_joint', 'wrist2_joint'))

# display frames
for node_name, node in node_dict.items():
    print(f"node_name: {node_name}")
    robot.viewer.gui.applyConfiguration(node_name, pin.SE3ToXYZQUATtuple(node))
robot.viewer.gui.refresh()
