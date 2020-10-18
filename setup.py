from distutils.core import setup
setup(
  name = 'DirectionalDiscrepancy',     
  packages = ['DirectionalDiscrepancy'],
  version = '1.1',
  license='MIT',
  description = 'Directional Discrepancy algorithm',
  author = 'Milad Bakhshizadeh, Mina Latifi, Ali Kamalinejad',
  author_email = 'mb4041@columbia.edu, minlat1375@gmail.com, kamalinejad.a@ut.ac.ir',
  url = 'https://github.com/minalatifi/DirectionalDiscrepancy',
  download_url = 'https://github.com/minalatifi/DirectionalDiscrepancy/archive/v1.1.tar.gz',
  keywords = ['Discrepancy', 'Cap Discerpancy', 'Directional Discrepancy', 'Polar Coordinates'],
  install_requires=[
          'mpmath',
          'numpy',
          'termcolor',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
     'Programming Language :: Python :: 3.7',
  ],
)
