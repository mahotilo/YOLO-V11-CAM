from setuptools import setup, find_packages

setup(
    name='yolo_cam',
    version='0.1',
    packages=find_packages(include=['yolo_cam', 'yolo_cam.*']),
    install_requires=[
        'ttach'
    ],
)
