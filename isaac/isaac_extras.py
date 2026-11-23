"""Inside Isaac Sim (installed by run_sim_62.py): ceiling lights where the hospital stage has none nearby, walking
people that follow /mantf/people (the same people the fleet model gets as other agents), and the kinematic teammates
as G1 bodies (the G1's visual meshes, no physics) at their poses from /mantf/fleet.

MANTF_ISAAC_LIGHTS=0 / MANTF_ISAAC_PEOPLE=0 switch either off. MANTF_LIGHT_BOX="x0,y0,x1,y1" limits the new lights."""
import math
import os
import sys
import threading
import time

import numpy as np

ASSETS = os.environ.get("ISAAC_ASSET_ROOT_60", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/6.0"))
CHARACTERS = ["Isaac/People/Characters/F_Medical_01/F_Medical_01.usd",
              "Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
              "Isaac/People/Characters/original_male_adult_medical_01/male_adult_medical_01.usd",
              "Isaac/People/Characters/original_female_adult_medical_01/female_adult_medical_01.usd",
              "Isaac/People/Characters/F_Business_02/F_Business_02.usd"]
WALK = "Isaac/People/Animations/stand_walk_loop_in_place.skelanim.usd"
ROOT = "/World/MantfPeople"
LIGHTS = "/World/MantfLights"
KIT_LUX_PER_M2, LIGHT_LEVEL = 130000 / 4.0 * 0.25, 0.12   # build_hospital.py
SPACING, LIGHT_Z, PANEL, NEAREST = 3.5, 2.85, 0.8, 3.0
PELVIS_Z = 0.79   # standing pelvis height of the G1


def log(msg):
    print(f"[mantf_extras] {msg}", flush=True)


def read_map(yaml_path):
    import yaml
    meta = yaml.safe_load(open(yaml_path))
    raw = open(os.path.join(os.path.dirname(yaml_path), meta["image"]), "rb").read()
    parts, i = [], 0
    while len(parts) < 4:
        while raw[i:i + 1].isspace():
            i += 1
        if raw[i:i + 1] == b"#":
            i = raw.index(b"\n", i)
            continue
        j = i
        while not raw[j:j + 1].isspace():
            j += 1
        parts.append(raw[i:j])
        i = j
    w, h = int(parts[1]), int(parts[2])
    img = np.frombuffer(raw[i + 1:i + 1 + w * h], np.uint8).reshape(h, w)
    occ = np.flipud(img < 128)
    return occ, float(meta["resolution"]), meta["origin"][:2]


class Extras:
    def __init__(self):
        import omni.kit.app
        self.want_lights = os.environ.get("MANTF_ISAAC_LIGHTS", "1") == "1"
        self.want_people = os.environ.get("MANTF_ISAAC_PEOPLE", "1") == "1"
        self.frames = 0
        self.lights_done = False
        self.node = None
        self.poses = None
        self.people = []
        self.mates = {}
        self.mate_poses = {}
        self._g1 = None
        self.walk = None
        self.lock = threading.Lock()
        self.sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.update)
        log(f"installed (lights {self.want_lights}, people {self.want_people})")

    def update(self, _e):
        try:
            import omni.usd
            ctx = omni.usd.get_context()
            stage = ctx.get_stage()
            if stage is None or ctx.get_stage_loading_status()[2] > 0 or not stage.GetPrimAtPath("/World/G1"):
                return
            self.frames += 1
            if self.frames < 20:
                return
            if self.want_lights and not self.lights_done:
                self.lights_done = True
                self.add_lights(stage)
            if self.want_people:
                if self.node is None:
                    self.start_ros()
                self.move_people(stage)
                self.move_mates(stage)
        except Exception as exc:
            log(f"update failed: {exc}")
            self.want_people = self.want_people and "rclpy" not in str(exc)

    # ------------------------------------------------------------- lights
    def add_lights(self, stage):
        from pxr import Gf, Usd, UsdGeom, UsdLux
        have = []
        for p in stage.Traverse():
            if p.HasAPI(UsdLux.LightAPI) and not p.IsA(UsdLux.DomeLight) and not p.IsA(UsdLux.DistantLight):
                t = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                have.append((t[0], t[1]))
        have = np.asarray(have, float).reshape(-1, 2)
        path = os.environ.get("MANTF_G1_MAP", "")
        if not os.path.isfile(path):
            log("no MANTF_G1_MAP: no extra lights")
            return
        occ, res, org = read_map(path)
        from scipy.ndimage import distance_transform_edt
        edt = distance_transform_edt(~occ) * res
        box = [float(v) for v in os.environ.get("MANTF_LIGHT_BOX", "").split(",")] if os.environ.get("MANTF_LIGHT_BOX") \
            else [org[0], org[1], org[0] + occ.shape[1] * res, org[1] + occ.shape[0] * res]
        UsdGeom.Scope.Define(stage, LIGHTS)
        n = 0
        for x in np.arange(box[0] + SPACING / 2, box[2], SPACING):
            for y in np.arange(box[1] + SPACING / 2, box[3], SPACING):
                r, c = int((y - org[1]) / res), int((x - org[0]) / res)
                if not (0 <= r < occ.shape[0] and 0 <= c < occ.shape[1]) or edt[r, c] < 0.6:
                    continue
                if len(have) and np.min(np.hypot(have[:, 0] - x, have[:, 1] - y)) < NEAREST:
                    continue
                light = UsdLux.RectLight.Define(stage, f"{LIGHTS}/L{n:04d}")
                light.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), LIGHT_Z))
                light.GetWidthAttr().Set(PANEL)
                light.GetHeightAttr().Set(PANEL)
                light.GetIntensityAttr().Set(KIT_LUX_PER_M2 * SPACING * SPACING / (PANEL * PANEL) * LIGHT_LEVEL)
                light.CreateEnableColorTemperatureAttr(True)
                light.CreateColorTemperatureAttr(4500.0)
                n += 1
        log(f"{len(have)} lights in the stage; added {n} ceiling lights where none was within {NEAREST} m")

    # ------------------------------------------------------------- people
    def start_ros(self):
        rclpy = sys.modules.get("rclpy")
        if rclpy is None or not rclpy.ok():
            return
        from geometry_msgs.msg import PoseArray
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        self.node = rclpy.create_node("mantf_isaac_people")
        self.node.create_subscription(PoseArray, "/mantf/people", self.on_people, qos_profile_sensor_data)
        from std_msgs.msg import String
        self.node.create_subscription(String, "/mantf/fleet", self.on_fleet, 10)
        ex = SingleThreadedExecutor()
        ex.add_node(self.node)
        threading.Thread(target=ex.spin, daemon=True).start()
        log("listening to /mantf/people")

    def on_people(self, msg):
        with self.lock:
            self.poses = [(p.position.x, p.position.y, math.atan2(2 * p.orientation.w * p.orientation.z,
                                                                   1 - 2 * p.orientation.z ** 2)) for p in msg.poses]

    def on_fleet(self, msg):
        import json
        d = json.loads(msg.data)
        with self.lock:
            self.mate_poses = {k: r["pose"] for k, r in d.get("robots", {}).items() if r.get("kind") == "kin" and r.get("pose")}

    def g1_visuals(self):
        """The G1's visual meshes (no physics) and their transforms relative to the pelvis, from WardZero's Isaac G1."""
        if self._g1 is None:
            from pxr import Gf, Usd, UsdGeom
            asset = os.path.join(os.environ.get("HG1_ROOT", ""), "g1-isaac-sim", "usd", "g1_29dof_brainco.usd")
            st = Usd.Stage.Open(asset)
            pelvis = st.GetPrimAtPath(f"{st.GetDefaultPrim().GetPath()}/Geometry/pelvis")
            inv = UsdGeom.Xformable(pelvis).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
            parts = []
            for q in Usd.PrimRange(pelvis):
                if q.GetTypeName() == "Mesh" and not any("Physics" in a for a in q.GetAppliedSchemas()):
                    # the mesh keeps its own local transform through the reference; the wrapper places its parent link
                    m = UsdGeom.Xformable(q).ComputeParentToWorldTransform(Usd.TimeCode.Default()) * inv
                    parts.append((str(q.GetPath()), m))
            self._g1 = (asset, parts)
            log(f"teammate body: {len(parts)} visual meshes of {asset}")
        return self._g1

    def move_mates(self, stage):
        """Kinematic teammates as G1 bodies (visual meshes only, standing pose), placed at their fleet poses."""
        from pxr import Gf, Sdf, UsdGeom
        with self.lock:
            poses = dict(self.mate_poses)
        cols = [(0.12, 0.47, 0.9), (0.98, 0.63, 0.12), (0.78, 0.24, 0.78), (0.24, 0.78, 0.24)]
        for k, (x, y, yaw) in poses.items():
            if k not in self.mates:
                asset, parts = self.g1_visuals()
                root = UsdGeom.Xform.Define(stage, f"/World/MantfTeammates/{k}")
                ops = (root.AddTranslateOp(), root.AddRotateZOp())
                for i, (path, m) in enumerate(parts):
                    xf = UsdGeom.Xform.Define(stage, f"/World/MantfTeammates/{k}/g1_{i:02d}")
                    xf.AddTransformOp().Set(m)
                    # untyped prim: the reference brings the Mesh type, geometry and local transform
                    stage.DefinePrim(f"/World/MantfTeammates/{k}/g1_{i:02d}/mesh").GetReferences().AddReference(
                        asset, Sdf.Path(path))
                disc = UsdGeom.Cylinder.Define(stage, f"/World/MantfTeammates/{k}/marker")
                disc.CreateRadiusAttr(0.38)
                disc.CreateHeightAttr(0.01)
                disc.AddTranslateOp().Set(Gf.Vec3d(0, 0, -PELVIS_Z + 0.006))
                disc.CreateDisplayColorAttr([cols[len(self.mates) % len(cols)]])
                self.mates[k] = ops
                log(f"teammate {k} shown as a G1")
            self.mates[k][0].Set(Gf.Vec3d(x, y, PELVIS_Z))
            self.mates[k][1].Set(math.degrees(yaw))

    def load_walk(self):
        from pxr import Usd, UsdSkel
        st = Usd.Stage.Open(os.path.join(ASSETS, WALK))
        anim = next(UsdSkel.Animation(p) for p in st.Traverse() if p.IsA(UsdSkel.Animation))
        times = anim.GetRotationsAttr().GetTimeSamples() or [0.0]
        self.walk = {"joints": list(anim.GetJointsAttr().Get()), "fps": st.GetTimeCodesPerSecond() or 30.0,
                     "rot": [anim.GetRotationsAttr().Get(t) for t in times],
                     "tr": [anim.GetTranslationsAttr().Get(t) for t in times],
                     "sc": [anim.GetScalesAttr().Get(t) for t in times]}
        log(f"walk cycle: {len(times)} frames, {len(self.walk['joints'])} joints")

    def spawn(self, stage, i):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel
        if self.walk is None:
            self.load_walk()
        path = f"{ROOT}/P{i}"
        xf = UsdGeom.Xform.Define(stage, path)
        xf.GetPrim().GetPayloads().AddPayload(os.path.join(ASSETS, CHARACTERS[i % len(CHARACTERS)]))
        stage.Load(path)
        xf.ClearXformOpOrder()
        ops = (xf.AddTranslateOp(), xf.AddRotateZOp())
        skel = next((q for q in Usd.PrimRange(xf.GetPrim()) if q.IsA(UsdSkel.Skeleton)), None)
        anim = None
        if skel is not None:
            anim = UsdSkel.Animation.Define(stage, skel.GetPath().AppendChild("MantfWalk"))
            anim.CreateJointsAttr(self.walk["joints"])
            UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
        self.people.append({"ops": ops, "anim": anim, "phase": float(i) * 7.0, "last": None, "t": time.monotonic()})
        log(f"person {i}: {CHARACTERS[i % len(CHARACTERS)]}")

    def move_people(self, stage):
        with self.lock:
            poses = self.poses
        if not poses:
            return
        from pxr import Gf
        while len(self.people) < len(poses):
            self.spawn(stage, len(self.people))
        now = time.monotonic()
        for p, (x, y, yaw) in zip(self.people, poses):
            dt, p["t"] = now - p["t"], now
            moving = p["last"] is not None and math.hypot(x - p["last"][0], y - p["last"][1]) > 1e-3
            p["last"] = (x, y)
            p["ops"][0].Set(Gf.Vec3d(x, y, 0.0))
            p["ops"][1].Set(math.degrees(yaw) + 90.0)
            if p["anim"] is not None:
                if moving:
                    p["phase"] += dt * self.walk["fps"]
                k = int(p["phase"]) % len(self.walk["rot"])
                p["anim"].GetRotationsAttr().Set(self.walk["rot"][k])
                p["anim"].GetTranslationsAttr().Set(self.walk["tr"][k])
                p["anim"].GetScalesAttr().Set(self.walk["sc"][k])


_extras = None


def install():
    global _extras
    if _extras is None:
        _extras = Extras()
    return _extras
