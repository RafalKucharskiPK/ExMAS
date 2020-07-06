from distutils.core import setup

setup(
    name='ExMAS',  # How you named your package folder (MyLib)
    packages=['ExMAS'],  # Chose the same as "name"
    version='0.9.99',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="Exact Matching of Attractive Shared rides (ExMAS) Kucharski R, Cats O TransResB, 2020 139 2020 285-310",
    author='Rafal Kucharski',  # Type in your name
    author_email='rkucharski@pk.edu.pl',  # Type in your E-Mail
    url='https://github.com/RafalKucharskiPK/ExMAS',  # Provide either the link to your github or to your website
    download_url='https://github.com/RafalKucharskiPK/ExMAS/archive/0.9.9.tar.gz',  # I explain this later on
    keywords=['MaaS', 'transportation', 'shared mobility'],  # Keywords that define your package best
    install_requires=['networkx>=2.4',
                      'pandas>=1.0.5',
                      'matplotlib>=3.2.2',
                      'dotmap>=1.2.20',
                      'osmnx>=0.15.0',
                      'PuLP>=1.6.8',
                      'numpy>=1.18.5'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3.7',
    ],
)
