from glob import glob

from setuptools import find_packages, setup

package_name = "dog_follow"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="bernard",
    maintainer_email="todo@example.com",
    description="Robot-agnostic RGB person following for legged robots",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "perception = dog_follow.nodes.perception_node:main",
            "controller = dog_follow.nodes.controller_node:main",
            "twist_adapter = dog_follow.nodes.robot_adapters:main_twist",
            "m20_adapter = dog_follow.nodes.robot_adapters:main_m20",
            "offline_demo = dog_follow.tools.offline_demo:main",
        ],
    },
)
