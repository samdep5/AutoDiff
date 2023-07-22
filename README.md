# Forward Automatic Differentiation Python Package Project
This is a automatic differentiation package that was created as a final project in CS107 by Sam DePaolo, Nick Dow, and Alexander Fung. How to install the package and documentation is located in "docs/documentation.ipynb". This package features forward automatic differentiation with some light backwards automatic differentiation (some parts of backwards autodiff were completed, but additional vector handling was not completed due to time constraints). This repository also features tests for our package. 

The package can be installed via the following commands:
'''
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ team48-autodiff-package
'''
Example use cases can be found in docs/documentation.ipynb.

# Broader Impact and inclusivity Statement

  **Broader Impact**
  
  Our software is intended to help people efficiently and conveniently find the derivatives and gradients of functions. Derivative calculators like Wolfram or graphing calculators are widely available, but an automatic differentiation package is particularly powerful because of its O(1) speed and ability to integrate into other programs. Our hope is that individuals will use this software in machine learning, game theory, mathematics, and other applicable fields to contribute meaningful, positive work to society as a whole. Misuses of this package could be as trivial as a high schooler cheating on calculus homework to working on serious ML research with the goal of personal game at the expense of others. As previously mentioned, our package has particular applications in AI, and significant and possible uses of our package could involve the creation of a model that can defeat any human at a game, which could be particularly problematic if an individual with a gambling addiction was pitted against such a model. One such example of this is the AlphaZero engine from chess:

  https://mmsubra1.medium.com/machine-learning-for-chess-alphazero-vs-stockfish-b58638e73fee

  While the latter example is exciting, it can easily be used by humans to cheat in online matches or even in in-person tournaments if a player smuggles in a device. With great computing power comes great responsibility, and we encourage our users to consider the consequences of their potential usage of our package.

  **Software Inclusivity**

  We encourage all people to use this software regardless of age, race, gender, socioeconomic status, national origin, or any other personal characteristics. That being said, we understand that the software is not equally accessible to everyone. For one thing, all of the comments, docstrings, command names, and this documentation file are in english or use english-based naming conventions. As the authors of this package, we are the most comfortable with English and didn’t have the resources or time to write multiple versions of the code. Fortunately, most of the actual interface only requires the user to read mathematical symbols and numbers. While the user will need to translate some minor documentation to understand how the AutoDiff object works, it is fairly straightforward and can be called without significant English knowledge.
