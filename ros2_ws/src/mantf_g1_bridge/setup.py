import os
from glob import glob

from setuptools import setup

package_name = "mantf_g1_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="shirish",
    maintainer_email="shirish@example.com",
    description="mantf planner and pick/place mission for the G1 in the WardZero hospital",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "fleet = mantf_g1_bridge.fleet_node:main",
            "pick_place = mantf_g1_bridge.pick_place_node:main",
            "walkers = mantf_g1_bridge.walkers_node:main",
            "goto = mantf_g1_bridge.goto:main",
            "sensor_viewer = mantf_g1_bridge.sensor_viewer:main",
            "kinematic_sim = mantf_g1_bridge.kinematic_sim:main",
        ],
    },
)
