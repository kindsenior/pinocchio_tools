import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

robot = load_robot_description("ur5_description")   # downloads/caches the UR5 URDF
model, data = robot.model, robot.data

# robot.setVisualizer(MeshcatVisualizer())
robot.setVisualizer(GepettoVisualizer())

# init gepetto-gui in another terminal
robot.initViewer(loadModel=True)
robot.display(q)

q  = pin.neutral(model)              # joint poisition
v  = 0.1 * pin.utils.rand(model.nv)  # joint velocity
a  = 0.2 * pin.utils.rand(model.nv)  # joint acceleration

# inverse dynamics
tau = pin.rnea(model, data, q, v, a)
print(f"tau:\n{tau}")

mtau = pin.crba(model, data, q) @ a
print(f"mtau:\n{mtau}")

ctau = pin.computeCoriolisMatrix(model, data, q, v) @ v
print(f"ctau:\n{ctau}")

gtau = pin.computeGeneralizedGravity(model, data, q)
print(f"gtau:\n{gtau}")

residual = tau - (mtau + ctau + gtau)
print(f"residual:\n{residual}")
print(f"|| residual ||={np.linalg.norm(residual)}")

# inverse dynamics with external forces
f_ext = pin.StdVec_Force()
f_ext.extend([ pin.Force.Zero() for _ in range(model.njoints) ])

## frame name -> joint id
# effort_frame_id = model.getFrameId("tool0")
# parent_id = model.frames[ee_frame_id].parentJoint

parent_id = model.getJointId('wrist_3_joint')  # parent joint id
local_placement  = pin.SE3.Identity()
local_placement.translation = np.array([0.0, 0.05, 0.1])
effort_frame_id = model.addFrame(
    pin.Frame(
        'contact_frame',
        parent_id,             # attach to wrist_3_link
        local_placement,     # fixed transform
        pin.FrameType.FIXED_JOINT
    )
)

# recreate data after addFrame
data = model.createData()
robot.data = data

# update data
robot.forwardKinematics(q)

# set force at the joint local frame
f_world = pin.Force(np.array([0,0,-9.81]), np.zeros(3)) # world frame
f_local = robot.data.oMf[parent_id].actInv(f_world)     # local frame
f_ext[parent_id] = pin.Force(f_local)

tau_with_forces = pin.rnea(model, data, q, v, a, f_ext)
print(f"tau with forces:\n {tau_with_forces}")

