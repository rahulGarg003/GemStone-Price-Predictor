from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n','') for req in requirements]
    
    if(HYPHEN_E_DOT in requirements):
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name = 'Gemstone-Price-Predictor',
    version = '0.0.1',
    author = 'Rahul Kumar Garg',
    install_requires = get_requirements('./requirements.txt'),
    packages=find_packages()
)
