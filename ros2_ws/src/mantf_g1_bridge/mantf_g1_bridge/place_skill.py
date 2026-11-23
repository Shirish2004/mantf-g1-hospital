from __future__ import annotations

import numpy as np

from g1_mission.geometry import inv
from g1_mission.skills.base import HAND_OPEN, Skill, Status, drive_arm


class PlaceOnSurface(Skill):
    name = "place"

    def start(self, ctx, instrument=None, surface_z=0.83, reach=None, **goal):
        super().start(ctx)
        self.inst = instrument or ctx.board.get("instrument")
        self.dt = 1.0 / self.p("rate", 50.0)
        self.surface_z = float(surface_z)
        ctx.arm.init(ctx.robot.upper_q())
        T = ctx.ik.fk(ctx.arm.q_des)
        base = ctx.robot.base_T()
        z_pelvis = self.surface_z - float(base[2, 3])
        T_to = ctx.board.get("T_tcp_obj")
        if T_to is not None:
            tip = T[:3, :3] @ np.asarray(T_to)[:3, 3]
        else:
            tip = T[:3, :3] @ np.array([0.0, 0.0, -float(ctx.catalog.get(self.inst, {}).get("grasp_z", 0.1))])
        tgt = T.copy()
        tgt[0, 3] = float(reach if reach is not None else self.p("reach", 0.40))
        tgt[2, 3] = z_pelvis + self.p("clearance", 0.012) - tip[2]
        self.T_down = tgt
        self.T_above = tgt.copy()
        self.T_above[2, 3] += self.p("above", 0.10)
        self.T_up = self.T_down.copy()
        self.T_up[2, 3] += self.p("rise", 0.12)
        self.T_up[0, 3] -= self.p("retreat", 0.06)
        self._go(ctx, "above")

    def _go(self, ctx, phase):
        now = ctx.robot.now()
        self.phase, self.phase_t = phase, now
        arm = ctx.arm
        if phase == "above":
            arm.move_cartesian(self.T_above, now, speed=0.10, q_rest=ctx.ready_pose)
        elif phase == "lower":
            arm.move_cartesian(self.T_down, now, speed=self.p("lower_speed", 0.04))
        elif phase == "release":
            arm.hold()
            arm.move_hand(arm.side, HAND_OPEN, now, self.p("open_time", 0.6))
        elif phase == "rise":
            arm.move_cartesian(self.T_up, now, speed=0.08)

    def tick(self, ctx):
        if drive_arm(ctx, self.dt) is None:
            return Status.RUNNING
        now = ctx.robot.now()
        arm = ctx.arm
        el = now - self.phase_t
        if self.phase in ("above", "lower") and arm.done(now) and arm.ik_err[0] > self.p("ik_tol", 0.03):
            return self.fail(f"{self.phase}: surface out of reach (IK error {arm.ik_err[0] * 100:.1f} cm)")
        if self.phase == "release":
            if el > self.p("open_time", 0.6) + self.p("settle_time", 0.6):
                self._go(ctx, "rise")
            return Status.RUNNING
        if not (arm.done(now) and el > arm.seg["dur"] + 0.3):
            return Status.RUNNING
        if self.phase == "above":
            self._go(ctx, "lower")
        elif self.phase == "lower":
            self._go(ctx, "release")
        elif self.phase == "rise":
            return self._verify(ctx)
        return Status.RUNNING

    def _verify(self, ctx):
        est = ctx.objects.get(self.inst) if (ctx.objects and self.inst) else None
        if est is None:
            self.message = "placed (not verified)"
            return Status.SUCCESS
        T_tcp = ctx.robot.base_T() @ ctx.ik.fk(ctx.robot.upper_q())
        d = float(np.linalg.norm(est.T[:3, 3] - T_tcp[:3, 3]))
        z = float(est.T[2, 3])
        if d < self.p("still_attached_dist", 0.05):
            return self.fail(f"{self.inst} still in the hand")
        if z < self.surface_z - self.p("drop_tol", 0.15):
            return self.fail(f"{self.inst} fell (z {z:.2f} m, surface {self.surface_z:.2f} m)")
        ctx.board.pop("T_tcp_obj", None)
        self.message = f"placed {self.inst} at z {z:.2f} m"
        return Status.SUCCESS
