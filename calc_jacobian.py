import numpy as np
import pinocchio as pin
from pinocchio.utils import *
from pinocchio.visualize import MeshcatVisualizer, GepettoVisualizer
from robot_wrapper import *

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
J = robot.getJointJacobian(ee_id, pin.ReferenceFrame.WORLD)
mprint(J)

# partial jacobian
keep_names = [
    "root_joint",
    "rarm_shoulder1_joint", "rarm_shoulder2_joint", "rarm_shoulder3_joint",
    "rarm_elbow_joint", "rarm_wrist1_joint", "rarm_wrist2_joint"
]
keep_ids = [model.getJointId(n) for n in keep_names]

# # from reducedModel
# lock_ids = [j.id for j in model.joints if j.id not in keep_ids]
# q_ref = pin.neutral(model)
# rmodel = pin.buildReducedModel(model, lock_ids, q_ref)
# robot_arm = RobotWrapper(rmodel)
# fid = rmodel.getFrameId("rarm_effector_body")
# q_arm = pin.neutral(rmodel)
# robot_arm.forwardKinematics(q_arm)
# J_arm = robot_arm.getFrameJacobian(fid, pin.ReferenceFrame.LOCAL)
# print("reduced Jacobian:", J_arm.shape) # (6,12)? not (6,6)?

# # extract jacobian of the full model
ee_frame = "rarm_effector_body"
fid = model.getFrameId(ee_frame)
start_end_joints = ('rarm_shoulder1_joint', 'rarm_wrist2_joint')
pos_cols = np.array(robot.get_position_index_list(*start_end_joints))
vel_cols = np.array(robot.get_velocity_index_list(*start_end_joints))
q = pin.neutral(model)
q[arm_cols+1] = np.deg2rad(np.array([0,-90,0,0,0,0]))
robot.forwardKinematics(q)
robot.display(q)
robot.computeJointJacobians(q)
J_frame = robot.getFrameJacobian(fid, pin.ReferenceFrame.WORLD)
J_arm = J_frame[:, vel_cols]
print(f"partial J for frame id:{ee_id}")
mprint(J_arm)
print("arm jacobian shape:", J_arm.shape) # (6, 6)
