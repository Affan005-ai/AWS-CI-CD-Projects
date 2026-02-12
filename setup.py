from setuptools import find_packages, setup  # pyright: ignore[reportMissingModuleSource]
from typing import List 

HYPNEN = "-e ."
def get_requirements(file_path: str) -> List[str]:
    """This function will return the list of requirements"""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPNEN in requirements:
            requirements.remove(HYPNEN)
    return requirements
 
setup(
    name="ML Projects",
    version="0.0.1",
    author="Affan",
    author_email="affannadeem005@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)