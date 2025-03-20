from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()

setup(
    name='file2data',
    version='0.1.1',
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            # Add your console scripts here
        ],
    },
    author='youdaoyzbx',
    author_email='youdaoyzbx@163.com',
    description='dataset loader tools for computer vision',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/youdaoyzbx/file2data',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
