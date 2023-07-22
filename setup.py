from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
    name='team48_autodiff_package',
    version='0.1.2',
    readme='README.md',
    license='MIT',
    author="Sam Dep Nick Dow Alex Fung",
    author_email='sdepaolo@college.harvard.edu',
    packages=['team48_autodiff_package.autoDiff', 'team48_autodiff_package.autoDiff.forward', 'team48_autodiff_package.autoDiff.reverse', 'team48_autodiff_package.dualNumber', 'team48_autodiff_package.nodeGraph'],
    include_package_data=True,
    install_requires=[],
)