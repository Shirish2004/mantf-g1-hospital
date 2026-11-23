import importlib
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
HG1 = os.environ.get("HG1_ROOT") or next((p for p in (os.path.join(_HERE, "..", "WardZero"), os.path.join(_HERE, "..", "..", "WardZero"))
                                          if os.path.isfile(os.path.join(p, "env.sh"))), os.path.join(_HERE, "..", "..", "WardZero"))
ASSETS = os.environ.get("ISAAC_ASSET_ROOT_60", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/6.0"))
EXTRA = [f"--/persistent/isaac/asset_root/default={ASSETS}",
         f"--/persistent/isaac/asset_root/cloud={ASSETS}",
         "--/app/extensions/fsWatcherEnabled=0"]
REQUIRED = ["omni.usd", "omni.timeline", "omni.kit.app", "omni.physx", "omni.physics.core", "omni.physics.tensors",
            "omni.physxfabric", "isaacsim.core.simulation_manager", "omni.replicator.core"]


def isaac_version():
    try:
        return open(os.path.join(os.environ.get("ISAAC_PATH", os.path.expanduser("~/isaacsim")), "VERSION")).read().strip()
    except OSError:
        return "?"


def patch_simulation_app():
    from isaacsim import SimulationApp
    orig = SimulationApp.__init__

    def init(self, launch_config=None, experience="", *a, **k):
        cfg = dict(launch_config or {})
        cfg["extra_args"] = list(cfg.get("extra_args", [])) + EXTRA + os.environ.get("ISAAC_EXTRA_ARGS", "").split()
        orig(self, cfg, experience, *a, **k)
        for name in ("isaacsim.core.simulation_manager", "omni.physics.tensors", "omni.physx.fabric"):
            try:
                from omni.kit.app import get_app
                get_app().get_extension_manager().set_extension_enabled_immediate(name, True)
            except Exception as exc:
                print(f"[run_sim_62] cannot enable {name}: {exc}", flush=True)
        if os.environ.get("MANTF_ISAAC_EXTRAS", "1") == "1":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import isaac_extras
            isaac_extras.install()

    SimulationApp.__init__ = init


def smoke():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    bad = []
    for m in REQUIRED:
        try:
            importlib.import_module(m)
        except Exception as exc:
            bad.append(f"{m}: {exc}")
    rclpy_dir = os.path.join(os.environ.get("ISAAC_DIR", os.path.expanduser("~/isaacsim")),
                             "exts/isaacsim.ros2.core/humble/rclpy")
    sys.path.insert(0, rclpy_dir)
    try:
        import rclpy
        rclpy.init()
        n = rclpy.create_node("isaac62_smoke")
        print(f"[run_sim_62] rclpy ok ({os.environ.get('RMW_IMPLEMENTATION')}), node {n.get_name()}", flush=True)
    except Exception as exc:
        bad.append(f"rclpy: {exc}")
    try:
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: F401
    except Exception as exc:
        bad.append(f"urdf importer: {exc}")
    print(f"[run_sim_62] Isaac {isaac_version()} python {sys.version.split()[0]}; "
          f"{'all APIs present' if not bad else 'MISSING: ' + '; '.join(bad)}", flush=True)
    app.close()
    os._exit(1 if bad else 0)


def main():
    args = sys.argv[1:]
    patch_simulation_app()
    if args and args[0] == "--smoke":
        smoke()
    target = os.path.join(HG1, "g1-isaac-sim", "run_sim.py")
    sys.argv = [target] + args
    print(f"[run_sim_62] Isaac {isaac_version()} running {target} {' '.join(args)} (asset root {ASSETS})", flush=True)
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
