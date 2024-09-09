from setuptools import setup, find_packages

setup(
    name='your-app-name',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'plotly',
        'streamlit',
        'streamlit-option-menu',
        'opencv-python-headless',
        'Pillow',
        'dlib',
        'face_recognition',
    ],
)
