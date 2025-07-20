import logging
# logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pinocchio as pin
from robot_wrapper import *

red   = [1,0,0, 0.7]
green = [0,1,0, 0.7]
blue  = [0,0,1, 0.7]

# generate robot model
model = pin.Model()
visual_model = pin.GeometryModel()
collision_model = visual_model.copy()
robot = RobotWrapper(model, collision_model, visual_model)
robot.setVisualizer(GepettoVisualizer())
# robot.setVisualizer(MeshcatVisualizer())

# Free-flyer root joint
joint_id = robot.model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root_joint")
robot.model.appendBodyToJoint(joint_id, pin.Inertia.Random(), pin.SE3.Identity())
frame_id = robot.model.addFrame(pin.Frame("root_link", joint_id, 0, pin.SE3.Identity(), pin.FrameType.BODY))
geom_obj = pin.GeometryObject(f"root_link_box", joint_id, frame_id, pin.SE3.Identity(), Box(0.2, 0.2, 0.1))
geom_obj.meshColor = np.array([0.8,0.8,0.8, 1.0])
robot.visual_model.addGeometryObject(geom_obj)
root_joint_id = joint_id

# rleg
parent_joint_id = root_joint_id
parent_joint_id, _ = robot.add_joint_link("rleg_joint1", "rleg_link1", [0,0,1],  [0,-0.1,-0.1], parent_joint_id, color=red)
parent_joint_id, _ = robot.add_joint_link("rleg_joint2", "rleg_link2", [1,0,0],  [0,0,-0.1],    parent_joint_id, color=green)
parent_joint_id, _ = robot.add_joint_link("rleg_joint3", "rleg_link3", [0,1,1],  [0,0,0],       parent_joint_id, link_translation=[0,0,-0.5], color=blue)
parent_joint_id, _ = robot.add_joint_link("rleg_joint4", "rleg_link4", [0,1,0],  [0,0,-0.5],    parent_joint_id, link_translation=[0,0,-0.5], color=red)
parent_joint_id, _ = robot.add_joint_link("rleg_joint5", "rleg_link5", [0,1,0],  [0,0,-0.5],    parent_joint_id, color=green)
parent_joint_id, _ = robot.add_joint_link("rleg_joint6", "rleg_link6", [1,0,0],  [0,0,0],       parent_joint_id, link_translation=[0.15,0,0], color=blue)

# lleg
parent_joint_id = root_joint_id
parent_joint_id, _ = robot.add_joint_link("lleg_joint1", "lleg_link1", [0,0,1],  [0,0.1,-0.1],  parent_joint_id, color=red)
parent_joint_id, _ = robot.add_joint_link("lleg_joint2", "lleg_link2", [1,0,0],  [0,0,-0.1],    parent_joint_id, color=green)
parent_joint_id, _ = robot.add_joint_link("lleg_joint3", "lleg_link3", [0,-1,1], [0,0,0],       parent_joint_id, link_translation=[0,0,-0.5], color=blue)
parent_joint_id, _ = robot.add_joint_link("lleg_joint4", "lleg_link4", [0,1,0],  [0,0,-0.5],    parent_joint_id, link_translation=[0,0,-0.5], color=red)
parent_joint_id, _ = robot.add_joint_link("lleg_joint5", "lleg_link5", [0,1,0],  [0,0,-0.5],    parent_joint_id, color=green)
parent_joint_id, _ = robot.add_joint_link("lleg_joint6", "lleg_link6", [1,0,0],  [0,0,0],       parent_joint_id, link_translation=[0.15,0,0], color=blue)

# initialize data
robot.q0 = np.zeros(robot.model.nq, dtype=np.double)
robot.rebuildData()

# visualize
robot.initViewer(loadModel=True)
# robot.initViewer(loadModel=True, open=True)
q = robot.q0.copy()
robot.display(q)

# inverse kinematics
def test_inverse_kinamtics():
    # target frame
    des_frame = pin.SE3(pin.rpy.rpyToMatrix(0,np.pi/6,0), np.array([0.3, -0.2, -0.6]))

    # add XYZaxis
    node_dict = {}
    axis_radius = 0.05; axis_size = 0.2
    node_name = "world/target"
    robot.viewer.gui.addXYZaxis(node_name, [1., 0., 0., 1.], axis_radius, axis_size)
    node_dict[node_name] = des_frame

    # q = robot.inverse_kinematics("wrist2_joint", des_frame)
    q = robot.inverse_kinematics("rleg_joint6", des_frame, robot.get_joint_names('rleg_joint1', 'rleg_joint6'))

    # display frames
    for node_name, node in node_dict.items():
        print(f"node_name: {node_name}")
        robot.viewer.gui.applyConfiguration(node_name, pin.SE3ToXYZQUATtuple(node))
    robot.viewer.gui.refresh()

# test_inverse_kinamtics()
