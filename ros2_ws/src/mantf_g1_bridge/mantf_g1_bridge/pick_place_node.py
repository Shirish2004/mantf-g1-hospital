from __future__ import annotations

import os

import py_trees
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from g1_mission import mission_node as mn
from g1_mission import skills as skill_lib
from g1_mission import tree as T
from g1_mission.config import load_mission, resolve

from .place_skill import PlaceOnSurface

skill_lib.SCRIPTED.setdefault("place", PlaceOnSurface)


class SetTask(T._Base):
    def __init__(self, name, core, instrument):
        super().__init__(name, core)
        self.instrument = instrument

    def update(self):
        c = self.core
        c.ctx.board.clear()
        c.ctx.board["instrument"] = self.instrument
        c.current = {"instrument": self.instrument, "t0": c.ctx.robot.now()}
        c.last_failure = ""
        return py_trees.common.Status.SUCCESS


class Fail(T._Base):
    def update(self):
        self.feedback_message = self.core.last_failure or "pick/place failed"
        return py_trees.common.Status.FAILURE


def build_pick_place(core, instrument, pick, place, home=""):
    surface = core.ctx.locations[place].get("surface_z", 0.83)
    reach = core.ctx.locations[place].get("reach")
    main = [SetTask("SetTask", core, instrument),
            T._go_and_align(core, f"ToPick({pick})", pick, "walk"),
            T.SkillBehaviour("ReAlignPick", core, "fine_align", {"station": core.ctx.locations[pick]}),
            T.SkillBehaviour("Locate", core, "locate"),
            T.SkillBehaviour("Pick", core, "pick"),
            T.SkillBehaviour("Carry", core, "carry"),
            T._go_and_align(core, f"ToPlace({place})", place, "hold"),
            T.SkillBehaviour("ReAlignPlace", core, "fine_align", {"station": core.ctx.locations[place]}),
            T.SkillBehaviour("Place", core, "place", {"surface_z": surface, "reach": reach}),
            T.SkillBehaviour("ReturnToReady", core, "return_to_ready")]
    if home and home in core.ctx.locations:
        main.append(T._go_and_align(core, f"Home({home})", home, "walk"))
    main.append(T.RecordResult("Done", core))
    recover = py_trees.composites.Sequence("Recover", memory=True, children=[
        T.RecordResult("LogFailure", core, success=False),
        T.SkillBehaviour("RecoverToReady", core, "return_to_ready"),
        Fail("Failed", core)])
    body = py_trees.composites.Selector("PickPlaceOrRecover", memory=True, children=[
        py_trees.composites.Sequence("PickPlace", memory=True, children=main), recover])
    return py_trees.composites.Sequence("pick_place", memory=True, children=[
        T.EnsureAuto("EnsureAuto", core), body, T.ReleaseAuto("ReleaseAuto", core)])


class PickPlaceNode(mn.MissionNode):
    def __init__(self):
        super().__init__()
        task = self.cfg.get("task", {})
        self.task = {"instrument": task.get("instrument", "scissors"),
                     "pick": task.get("pick", "theatre_instrument_table"),
                     "place": task.get("place", "pre_op_desk"),
                     "home": task.get("home", "")}
        self.locations.update(self.cfg.get("places", {}))
        self.ctx.locations = self.locations
        self.get_logger().info(f"pick_place ready: {self.task}; command 'pickplace[:instrument[:place[:pick]]]'")

    def on_command(self, m):
        cmd = m.data.strip()
        if cmd == "start" or cmd.startswith("pickplace"):
            parts = [p.strip() for p in cmd.split(":")[1:]]
            task = dict(self.task)
            for key, val in zip(("instrument", "place", "pick", "home"), parts):
                if val:
                    task[key] = val
            bad = [task[k] for k in ("pick", "place") if task[k] not in self.locations]
            if task["instrument"] not in self.catalog:
                bad.append(task["instrument"])
            with self.lock:
                if bad:
                    self.message = f"unknown {bad}; locations: {', '.join(self.locations)}"
                else:
                    self.pending = task
                    self.start("pickplace")
            self.publish_status()
            return
        super().on_command(m)

    def start(self, kind):
        if kind == "scrub_nurse":
            self.pending, kind = dict(self.task), "pickplace"
        if kind != "pickplace":
            return super().start(kind)
        if self.state in ("running", "paused"):
            self.message = f"already running {self.kind}"
            return
        task = getattr(self, "pending", None) or self.task
        self.ctx.arm.reset()
        self.ctx.board.clear()
        self.root = build_pick_place(self.core, **task)
        self.root.setup_with_descendants()
        self.kind = f"pickplace:{task['instrument']}:{task['pick']}->{task['place']}"
        self.state, self.message = "running", f"{self.kind} started"
        self.auto_seen = self.mode == T.AUTO
        self.get_logger().info(self.message)


def main():
    rclpy.init()
    node = PickPlaceNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
