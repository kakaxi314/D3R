from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='PSC',
    ext_modules=[
        CUDAExtension('PSC', [
            'psc.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })