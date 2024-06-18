from setuptools import setup, find_packages

setup(
    name = 'hu-coca',
    packages = find_packages(exclude = []),
    version = '0.0.1',
    license = 'MIT',
    description = 'MangaliCa - Hungarian CoCa model - PyTorch',
    author = 'kevinyecs',
    author_email = '',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/kevinyecs/hu-coca',
    keywords = [
        'artificial intelligence',
        'deep learning',
        'transformers',
        'attention mechanism',
        'vision transformer',
        'fourier transformation',
        'latent image representation'
    ],
    install_requires = [
        'einops'
    ],
    classifiers = [
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: MIT License',
        'Programming Language :: Python :: 3'
    ]
)