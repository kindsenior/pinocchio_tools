import numpy as np
import pinocchio as pin
from numpy.linalg import norm, solve
# from pinocchio.robot_wrapper import RobotWrapper
# from pinocchio.visualize import MeshcatVisualizer, GepettoVisualizer
from robot_wrapper import *

model = pin.buildSampleModelManipulator()
visual_model  = pin.buildSampleGeometryModelManipulator(model)
collision_model = visual_model.copy()
robot = RobotWrapper(model, collision_model, visual_model)
robot.setVisualizer(GepettoVisualizer())
robot.initViewer(loadModel=True)
q = robot.q0
robot.display(q)

JOINT_ID = 6
des_frame = pin.SE3(np.eye(3), np.array([1.0, 0.0, 1.0]))

# add XYZaxis
node_dict = {}
axis_radius = 0.05; axis_size = 0.2
node_name = "world/target"
robot.viewer.gui.addXYZaxis(node_name, [1., 0., 0., 1.], axis_radius, axis_size)
node_dict[node_name] = des_frame

eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12

i = 0
while True:
    pin.forwardKinematics(robot.model, robot.data, q)
    iMd = robot.data.oMi[JOINT_ID].actInv(des_frame)
    err = pin.log(iMd).vector  # in joint frame
    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    J = pin.computeJointJacobian(robot.model, robot.data, q, JOINT_ID)  # in joint frame
    J = -np.dot(pin.Jlog6(iMd.inverse()), J)
    v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pin.integrate(model, q, v * DT)
    if not i % 10:
        print(f"{i}: error = {err.T}")
    i += 1

if success:
    print("Convergence achieved!")
else:
    print(
        "\n"
        "Warning: the iterative algorithm has not reached convergence "
        "to the desired precision"
    )

robot.display(q)
# display frames
for node_name, node in node_dict.items():
    print(f"node_name: {node_name}")
    robot.viewer.gui.applyConfiguration(node_name, pin.SE3ToXYZQUATtuple(node))
robot.viewer.gui.refresh()

print(f"\nresult: {q.flatten().tolist()}")
print(f"\nfinal error: {err.T}")
# robot.add_fixed_frame_with_axis("force_local", "wrist_3_joint", offset_pos=(0, 0, 0.1))
# robot.display_with_frames(q)
