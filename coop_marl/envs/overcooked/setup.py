from setuptools import find_packages, setup

setup(
    name="cooking-gym",
    version="0.0.1",
    description='Cooking gym with graphics and ideas based on: "Too Many Cooks: Overcooked environment"',
    packages=find_packages() + [""],
    package_data={
        "gym_cooking": [
            "utils/new_style_level/*.json",
            "misc/game/graphics/*.png",
        ],
    },
    install_requires=[],
)