from setuptools import setup, find_packages

setup(
    name='lifelike',
    version='1.0.6',
    description='A A toolkit that allows for the creation of "lifelike" characters that you can interact with and change how they behave towards you',
    author='Mustafa Tariq and Khoa Nguyen',
    license='MIT',
    packages=find_packages(),
        install_requires=[
        'chromadb',
        'langchain'
    ]
)