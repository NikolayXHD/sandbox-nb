Построить набор данных для регрессии индикатор - изменение цены.

Идея в том, чтобы сравнивать цену в t и t+delta, где delta фиксирована.
Мы не боремся с волатильностью цены путем агрегации цены в будущем, с идеей,
что при построении регрессии шум всё равно будет устранён.

Таким образом поборены проблемы
- дорогостоящих агрегаций при построении набора данных
- размер набора данных для регрессии пропорционален количеству свеч без
  больших множителей

1 наблюдение = (
  ticker
  t_now
  t_delta
  indicator_1  # обычный cmf
  ...
  indicator_n  # вариации cmf с другим временем, степенью

  some_indicator(volume * price)  # Надежда на то, что объём торгов позволит отсеять мусорные акции
  ln(price_delta_relative)  # ln для аддитивности
)

ticker пробегает потенциально все ценные бумаги
delta пробегает характерные интервалы
1h, 1d, 1w, 1m, 2m, 4m, 8m

- для небольших интервалов < 1w отбрасывание случаев, когда t+delta приходится
  на выходной может исказить данные. Писать в величину t_delta_actual честное расстояние
  до ближайшего t2 >= t1 + delta. Регрессия покажет, что лучше предсказывает изменение,
  t_delta или t_delta_actual

infrastructure
==============

run tests ::

 make test

run tests with custom flags ::

 make test-vv
 make test-s

run linters ::

 make lint

setup jupyter
-------------

theme
~~~~~
jt -t oceans16 -T -N -f firacode -fs 13 -cellw 95% -lineh 120

conda
~~~~~

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
