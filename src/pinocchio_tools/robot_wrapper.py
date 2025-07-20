from hppfcl import Box, Cylinder
import logging
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
        logging.debug(f"node_name: {node_name}")
        self.viewer.gui.applyConfiguration(node_name, pin.SE3ToXYZQUATtuple(self.data.oMf[frame_id]))
    self.viewer.gui.refresh()

# robot modeling
def add_joint_link(self, joint_name, link_name, joint_axis, joint_translation, parent_joint_id, link_translation = None, color = None):
    """add a joint and its link frames

    Ags:
    joint_name(str)
    link_name(str)
    joint_axis(list)
    joint_translation(list)
    parent_joint_id(int)
    link_translation(list): the vector depicts the link shape
    color(list)

    Returns:
    joint_id(int), frame_id(int)
    """
    if type(joint_axis) == list: joint_axis = np.array(joint_axis, dtype=np.double)
    joint_axis /= np.linalg.norm(joint_axis)

    if type(joint_translation) == list: joint_translation = np.array(joint_translation, dtype=np.double)
    joint_placement = pin.SE3(np.eye(3), joint_translation)

    if type(link_translation) == list: link_translation = np.array(link_translation, dtype=np.double)

    if color is None: color = [0.8, 0.8, 0.8, 1.0]
    if type(color) == list: color = np.array(color, dtype=np.float32)

    joint_radius = 0.05 # the radius of the joint cylinder
    joint_width = 0.1   # the width of the joint cylinder

    z_axis = np.array([0,0,1], dtype=np.double)

    # add joint
    joint_id = self.model.addJoint(parent_joint_id, pin.JointModelRevoluteUnaligned(joint_axis), joint_placement, joint_name)
    self.model.appendBodyToJoint(joint_id, pin.Inertia.Random(), pin.SE3.Identity())

    # add frame
    frame_id = self.model.addFrame(pin.Frame(link_name, joint_id, 0, pin.SE3.Identity(), pin.FrameType.BODY))

    def generate_axial_geometry(default_axis, target_axis, geometry_size, joint_id, frame_id, geometry_name):
        """generate a geometry (like a cylinder or a link-shaped box) aligned the the target_axis

        Ags:
        default_axis(numpy.array): the default height axis direction of the geometry (e.g. cylinder's default_axis is z-axis [0,0,1])
        target_axis (numpy.array): the target direction to align the geometry
        geometry_size(tupple)
        joint_id(int)
        frame_id(int)
        geometry_name(str)

        Returns:
        pinocchio.pinocchio_pywrap_default.GeometryObject: the generated geometry object
        """
        v = np.cross(default_axis, target_axis)
        # rotate the geometry by the angle axis of default_axis x target_axis
        rotation = pin.AngleAxis(np.arccos(default_axis @ target_axis), v/np.linalg.norm(v)).matrix() if np.linalg.norm(v) > np.finfo(np.float32).eps else np.eye(3)
        geom_obj = pin.GeometryObject(geometry_name,
                                      joint_id,
                                      frame_id,
                                      pin.SE3(rotation, np.zeros(3, dtype=np.float32)),
                                      Cylinder(*geometry_size),
        )
        logging.info(f"rotation:\n{rotation}")
        return geom_obj

    # visual geometries
    ## add a cylinder for rotational joints
    geom_obj = generate_axial_geometry(z_axis, joint_axis, (joint_radius, joint_width), joint_id, frame_id, f"{link_name}_joint")
    geom_obj.meshColor = color
    self.visual_model.addGeometryObject(geom_obj)

    ## add a geometry of the link shape
    if link_translation is not None:
        link_direction = link_translation/np.linalg.norm(link_translation)
        # the geometry's name must be different with the joint's geometry's
        geom_obj = generate_axial_geometry(z_axis, link_direction,
                                           (joint_radius*0.5, np.linalg.norm(link_translation)),
                                           joint_id, frame_id, f"{link_name}_shape",
        )
        geom_obj.placement.translation = 0.5*link_translation
        geom_obj.meshColor = color
        self.visual_model.addGeometryObject(geom_obj)

    logging.info(f"parent_joint_id:{parent_joint_id}")
    logging.info(f"joint_id:{joint_id}")
    logging.info(f"frame_id:{frame_id}")

    return joint_id, frame_id

# kinematics
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

def get_joint_jacobian(self, joint_angles, target_joint, joint_names = None, frame=pin.ReferenceFrame.WORLD):
    joint_id = self.model.getJointId(target_joint)

    # set joint angles for FK
    if len(joint_angles) == len(self.q0):
        q = joint_angles
    else:
        q = np.zeros_like(self.q0)
        q[self.get_position_index_list(joint_names = joint_names)] = joint_angles

    # ForwardKinematics
    self.forwardKinematics(q)
    self.computeJointJacobians(q)

    # calculate Jacobian
    J = self.getJointJacobian(joint_id, frame)
    if joint_names is None:
        return J
    else:
        vel_cols = self.get_velocity_index_list(joint_names = joint_names)
        return J[:, vel_cols]

def normalize_pi(self, angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def inverse_kinematics(self, target_joint_name, target_frame,
                       joint_names = None,
                       init_q = None,
                       damp = 1e-12,
                       DT = 1e-1,
                       eps = 1e-4,
                       IT_MAX = 1000,
                       visualize = True):
    q = self.q0.copy() if init_q is None else init_q
    if target_joint_name in self.model.names:
        joint_id = self.model.getJointId(target_joint_name)
    else:
        logging.error(f"{target_joint_name} is not included in {self.model.names.tolist()}")
        return q

    is_partial = False
    if  joint_names is not None:
        pos_cols = np.array(self.get_position_index_list(joint_names = joint_names))
        vel_cols = np.array(self.get_velocity_index_list(joint_names = joint_names))
        is_partial = True

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
        if is_partial:
            J = J[:,vel_cols]
            v = -J.T @ solve(J.dot(J.T) + damp * np.eye(6), err)
            # q[pos_cols] = pin.integrate(self.model, q[pos_cols], v * DT) # The configuration vector is not of the right size
            q[pos_cols] += v * DT
        else:
            v = -J.T @ solve(J.dot(J.T) + damp * np.eye(6), err)
            q = pin.integrate(self.model, q, v * DT)

        q = self.normalize_pi(q) # normalize joint angles into -pi to pi
        if not i % 10:
            logging.debug(f"{i}: error = {err.T}")
            i += 1

    if success:
        logging.debug("Convergence achieved!")
    else:
        logging.error(
            "\n"
            "Warning: the iterative algorithm has not reached convergence "
            "to the desired precision"
        )
    logging.debug(f"\nresult: {q.flatten().tolist()}")
    logging.debug(f"\nfinal error: {err.T}")

    if visualize: self.display(q)

    return q

# monkey-patch to RobotWrapper
from pinocchio.robot_wrapper import RobotWrapper
# add method after import
RobotWrapper.add_fixed_frame_with_axis = add_fixed_frame_with_axis
RobotWrapper.display_with_frames       = display_with_frames
RobotWrapper.add_joint_link            = add_joint_link
RobotWrapper.get_joint_jacobian        = get_joint_jacobian
RobotWrapper.get_joint_names           = get_joint_names
RobotWrapper.get_position_index_list   = get_position_index_list
RobotWrapper.get_velocity_index_list   = get_velocity_index_list
RobotWrapper.normalize_pi              = normalize_pi
RobotWrapper.inverse_kinematics        = inverse_kinematics
