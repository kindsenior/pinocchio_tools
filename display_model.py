import numpy as np
import pinocchio as pin
from pathlib import Path
# from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer, GepettoVisualizer
from robot_descriptions.loaders.pinocchio import load_robot_description

# add a frames and XYZaxis
def add_fixed_frame_with_axis(self, name, parent_joint_name,
                              offset_pos=(0,0,0), offset_rpy=(0,0,0),
                              axis_radius=0.01, axis_size=0.05):
    # add a frame
    parent_id  = self.model.getJointId(parent_joint_name)
    local_offset  = pin.SE3(pin.rpy.rpyToMatrix(*offset_rpy), np.array(offset_pos))
    frame_id = self.model.addFrame(pin.Frame(name, parent_id, local_offset, pin.FrameType.FIXED_JOINT))
    self.rebuildData()

    # add an XYZaxis
    if name not in getattr(self, "_axis_nodes", {}):
        node_name = f"world/{name}"
        self.viewer.gui.addXYZaxis(node_name, [1., 0., 0., 1.], axis_radius, axis_size)

        self._axis_nodes = getattr(self, "_axis_nodes", {})
        self._axis_nodes[frame_id] = node_name
    return frame_id

# FK + frameFK + update frames
def display_with_frames(self, q):
    # pin.forwardKinematics(self.model, self.data, q)
    # pin.updateFramePlacements(self.model, self.data)
    # pin.updateGeometryPlacements(self.model, self.data, self.viz.visual_model, self.viz.visual_data)
    # self.viz.display()
    self.viz.display(q)
    pin.updateFramePlacements(self.model, self.data)

    for frame_id, node_name in getattr(self, "_axis_nodes", {}).items():
        # axis_name = f"world/{node_name}"
        print(f"node_name: {node_name}")
        self.viewer.gui.applyConfiguration(node_name, pin.SE3ToXYZQUATtuple(self.data.oMf[frame_id]))
    self.viewer.gui.refresh()
    # self.viz.display(q)

# monkey-patch to RobotWrapper
from pinocchio.robot_wrapper import RobotWrapper
RobotWrapper.add_fixed_frame_with_axis = add_fixed_frame_with_axis
RobotWrapper.display_with_frames       = display_with_frames

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
