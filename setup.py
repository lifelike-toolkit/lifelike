from setuptools import setup

setup(
    name='lifelike',
    version='1.0.6',
    description='A A toolkit that allows for the creation of "lifelike" characters that you can interact with and change how they behave towards you',
    author='Mustafa Tariq and Khoa Nguyen',
    license='MIT',
    packages=['lifelike'],
    package_dir={'': 'src'},
        install_requires=[
        'requests',
        'openai',
        'streamlit',
        'flask',
        'torch',
        'transformers',
        'onnx',
        'onnxruntime',
        'datasets',
        'chromadb',
        'flask-cors',
        'langchain'
    ]
)