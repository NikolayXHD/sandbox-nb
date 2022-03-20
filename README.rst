run tests ::

 make test

run tests with custom flags ::

 make test-vv
 make test-s

run linters ::

 make lint

setup jupyter
=============

theme
-----
jt -t oceans16 -T -N -f firacode -fs 13 -cellw 95% -lineh 120

conda
=====

https://conda.io/projects/conda/en/latest/user-guide/install/linux.html
https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf
https://rapids.ai/start.html

setup fish shell
----------------

.. code-block::

   ~/miniconda3/condabin/conda init fish


create environment
------------------

https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

.. code-block::

   conda env create -p ./envs -f environment.yml


setup environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

e.g. PYTHONPATH, see

.. code-block::

   ./envs/etc/conda/activate.d/env_vars.sh
   ./envs/etc/conda/deactivate.d/env_vars.sh


activate environment
--------------------

Do not use fish, because https://github.com/conda/conda/issues/7993
Run from e.g. bash shell

.. code-block::

   conda activate ./envs


alternatively, run command in environment context by

.. code-block::

   conda run -p ./envs command to run
