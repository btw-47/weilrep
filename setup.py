from setuptools import setup

setup(
   name='weilrep',
   version='1.0',
   description='A useful module',
   author='Brandon Williams',
   author_email='btw@math.berkeley.edu',
   packages=['weilrep'],  #same as name
   install_requires=['sage'], #external packages as dependencies
)