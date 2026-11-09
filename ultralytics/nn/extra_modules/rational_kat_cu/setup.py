import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


sources = glob.glob('src/*.cpp')+glob.glob('src/*.cu')


setup(
    name='kat_rational',  
    version='0.4',  
    author='adamdad',  
    author_email='yxy_adadm@qq.com',  
    description='A simple example of a PyTorch extension, implementing a group-wise rational function for kat',  
    long_description="""This package provides a PyTorch extension for computing group-wise rational functions. 
                        It is designed to be used as part of the 'kat' project, enhancing its capabilities in handling
                        specialized mathematical functions with optimized CUDA support.""",  
    
    
    
    
    
    cmdclass={'build_ext': BuildExtension}
)