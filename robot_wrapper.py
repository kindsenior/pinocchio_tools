import numpy as np
from numpy.linalg import norm, solve
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer, GepettoVisualizer

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
    ## update viz.data and viz.visual_data for visualizing
    ## data and visual_data are different b/w robot and viz
    # pin.forwardKinematics(self.model, self.viz.data, q)
    # pin.updateGeometryPlacements(self.model, self.viz.data, self.viz.visual_model, self.viz.visual_data)
    # self.viz.display()
    self.display(q)
    self.framesForwardKinematics(q)

    for frame_id, node_name in getattr(self, "_axis_nodes", {}).items():
        print(f"node_name: {node_name}")
        self.viewer.gui.applyConfiguration(node_name, pin.SE3ToXYZQUATtuple(self.data.oMf[frame_id]))
    self.viewer.gui.refresh()

def get_joint_names(self, start_joint, end_joint):
    """get a joint names' list from the start joint name and the end joint

    Ags:
    arg1(str): the joint name of the start of the list
    arg2(str): the joint name of the end of the list

    Returns:
    list: the list including the joint names
    """
    joint_names = self.model.names.tolist()
    start = joint_names.index(start_joint)
    end   = joint_names.index(end_joint)
    return joint_names[start : end + 1]

def get_position_index_list(self, start_joint = None, end_joint = None, joint_names = None):
    if joint_names is None:
        joint_names = self.get_joint_names(start_joint, end_joint)
    q_cols = []
    for jname in joint_names:
        jid  = self.model.getJointId(jname)
        q_cols.extend(range(self.model.idx_qs[jid], self.model.idx_qs[jid] + self.model.nqs[jid]))
    return q_cols

def get_velocity_index_list(self, start_joint = None, end_joint = None, joint_names = None):
    if joint_names is None:
        joint_names = self.get_joint_names(start_joint, end_joint)
    v_cols = []
    for jname in joint_names:
        jid  = self.model.getJointId(jname)
        v_cols.extend(range(self.model.idx_vs[jid], self.model.idx_vs[jid] + self.model.nvs[jid]))
    return v_cols

def inverse_kinematics(self, joint_name, target_frame,
                       init_q = None,
                       damp = 1e-12,
                       DT = 1e-1,
                       eps = 1e-4,
                       IT_MAX = 1000,
                       visualize = True):
    q = self.q0 if init_q is None else init_q
    if joint_name in self.model.names:
        joint_id = self.model.getJointId(joint_name)
    else:
        print(f"{joint_name} is not included in {self.model.names.tolist()}")
        return q

    i = 0
    while True:
        pin.forwardKinematics(self.model, self.data, q)
        iMd = self.data.oMi[joint_id].actInv(target_frame)
        err = pin.log(iMd).vector  # in joint frame

        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pin.computeJointJacobian(self.model, self.data, q, joint_id)  # in joint frame
        J = -pin.Jlog6(iMd.inverse()) @ J
        v = -J.T @ solve(J.dot(J.T) + damp * np.eye(6), err)
        q = pin.integrate(self.model, q, v * DT)
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
    print(f"\nresult: {q.flatten().tolist()}")
    print(f"\nfinal error: {err.T}")

    if visualize: self.display(q)

    return q

# monkey-patch to RobotWrapper
from pinocchio.robot_wrapper import RobotWrapper
# add method after import
RobotWrapper.add_fixed_frame_with_axis = add_fixed_frame_with_axis
RobotWrapper.display_with_frames       = display_with_frames
RobotWrapper.get_joint_names           = get_joint_names
RobotWrapper.get_position_index_list   = get_position_index_list
RobotWrapper.get_velocity_index_list   = get_velocity_index_list
RobotWrapper.inverse_kinematics        = inverse_kinematics
