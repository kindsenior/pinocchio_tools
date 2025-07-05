import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer, GepettoVisualizer

model = pin.buildSampleModelHumanoid()
visual_model  = pin.buildSampleGeometryModelHumanoid(model)
collision_model = visual_model.copy()
robot = RobotWrapper(model, collision_model, visual_model)
robot.setVisualizer(GepettoVisualizer())
# robot.setVisualizer(MeshcatVisualizer())
robot.initViewer(loadModel=True)
q = robot.q0
robot.display(q)

# calc jacobian from joints
robot.forwardKinematics(q)
robot.computeJointJacobians(q)
ee_id = robot.model.getJointId("rarm_wrist2_joint")
J = robot.getJointJacobian(ee_id, pin.ReferenceFrame.LOCAL)
print(J.shape)
