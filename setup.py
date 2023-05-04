import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='kode_search',
    version='0.0.1',
    author='Feng',
    author_email='feng@exp.ai',
    description='Search large code base with natural language questions.',
    install_requires=[
                'annoy==1.17.2',
                'faiss==1.7.2',
		        'gitpython==3.1.31',
                'numpy==1.24.2',
                'openai==0.27.4',
                'sentence-transformers==2.2.2',
                'setuptools==67.7.2',
                'tiktoken==0.3.3',
                'torch==2.0.0',
                'tree_sitter==0.20.1',
                'tree_sitter_builds==2023.3.12',
                'tree_sitter_languages==1.5.0',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    url='https://github.com/engianx/kode-search',
    package_dir={
        "kode_search": "src/kode_search"
    },
    package_data={
        "kode_search": ["templates/*", "static/css/*"]
    },
    entry_points={
        'console_scripts': [
            'kcs=kode_search.main:main',
        ]
    },
    keywords='code search, kode search',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development',
        'Topic :: Utilities',
    ]
)
