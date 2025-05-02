import sapien
import numpy as np
import warnings
import os


class XArm7(sapien.Widget):
    def __init__(self):
        pass

    def load(self, scene: sapien.Scene):
        if not scene.physx_system.config.enable_tgs:
            warnings.warn(
                "TGS is not enabled in scene. TGS is recommended for simulating loop joints."
            )
        # if scene.physx_system.config.solver_iterations < 15:
        #     warnings.warn(
        #         f"Solver iteration ({scene.physx_system.config.solver_iterations}) of this sceen is probably too small for simulating XArm"
        #     )

        loader = scene.create_urdf_loader()
        loader.set_material(0.3, 0.3, 0.0)
        loader.set_link_material("left_finger", 1.0, 1.0, 0.0)
        loader.set_link_material("right_finger", 1.0, 1.0, 0.0)
        loader.set_link_patch_radius("left_finger", 0.05)
        loader.set_link_patch_radius("right_finger", 0.05)
        loader.set_link_min_patch_radius("left_finger", 0.05)
        loader.set_link_min_patch_radius("right_finger", 0.05)

        path = os.path.join(os.path.dirname(__file__), "xarm_urdf/xarm7_gripper.urdf")
        self.robot = loader.load(path)
        for link in self.robot.links:
            link.disable_gravity = True

        self._create_drives(scene)

        self.arm_joints = self.robot.active_joints[:7]
        self.left_gripper_joint = self.robot.find_link_by_name(
            "left_outer_knuckle"
        ).joint
        self.right_gripper_joint = self.robot.find_link_by_name(
            "right_outer_knuckle"
        ).joint

        self.set_arm_pd([200] * 7, [20] * 7, [38] * 7)
        self.set_gripper_pd(1e4, 1e2, 0.3)

        self.robot.set_qpos([0, 0, 0, np.pi / 3, 0, np.pi / 3, -np.pi / 2] + [0] * 6)
        self.robot.set_qvel(np.zeros(13))
        self.set_arm_target([0, 0, 0, np.pi / 3, 0, np.pi / 3, -np.pi / 2])
        self.set_gripper_target(0)

    def _create_drives(self, scene):
        self.robot.set_qpos(np.zeros(self.robot.dof))

        lik = self.robot.find_link_by_name("left_inner_knuckle")
        lok = self.robot.find_link_by_name("left_outer_knuckle")
        lf = self.robot.find_link_by_name("left_finger")
        T_pw = lf.pose.inv().to_transformation_matrix()
        p_w = lf.pose.p + lik.pose.p - lok.pose.p
        T_fw = lik.pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]
        drive = scene.create_drive(lik, sapien.Pose(p_f), lf, sapien.Pose(p_p))
        drive.set_limit_x(0, 0)
        drive.set_limit_y(0, 0)
        drive.set_limit_z(0, 0)

        rik = self.robot.find_link_by_name("right_inner_knuckle")
        rok = self.robot.find_link_by_name("right_outer_knuckle")
        rf = self.robot.find_link_by_name("right_finger")
        T_pw = rf.pose.inv().to_transformation_matrix()
        p_w = rf.pose.p + rik.pose.p - rok.pose.p
        T_fw = rik.pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]
        drive = scene.create_drive(rik, sapien.Pose(p_f), rf, sapien.Pose(p_p))
        drive.set_limit_x(0, 0)
        drive.set_limit_y(0, 0)
        drive.set_limit_z(0, 0)

        gear = scene.create_gear(lok, sapien.Pose(), rok, sapien.Pose(q=[0, 0, 0, 1]))
        gear.gear_ratio = -1
        gear.enable_hinges()

        for l in [lik, lok, lf, rik, rok, rf]:
            for s in l.collision_shapes:
                s.set_collision_groups([1, 1, 2, 0])

    def set_arm_pd(self, ps, ds, limits):
        for j, p, d, l in zip(self.arm_joints, ps, ds, limits):
            j.set_drive_property(p, d, l, "acceleration")

    def set_arm_target(self, target):
        for t, j in zip(target, self.arm_joints):
            j.set_drive_target(t)

    def set_arm_target_velocity(self, target):
        for t, j in zip(target, self.arm_joints):
            j.set_drive_velocity_target(t)

    def set_gripper_pd(self, p, d, limit):
        lok = self.robot.find_link_by_name("left_outer_knuckle")
        rok = self.robot.find_link_by_name("right_outer_knuckle")
        self.left_gripper_joint.set_drive_property(p, d, limit, "acceleration")
        self.right_gripper_joint.set_drive_property(p, d, limit, "acceleration")

    def set_gripper_target(self, target):
        self.left_gripper_joint.set_drive_target(target)
        self.right_gripper_joint.set_drive_target(target)

    def unload(self, scene: sapien.Scene):
        scene.remove_articulation(self.robot)
