Installation
============

A modern NVIDIA GPU is required to run this application.

Anything less than 8GB of VRAM is not recommended, even that might be too little.

Multiple models are loaded into memory, so the more VRAM you have, the more models you can load.

On a 16gb GPU, you can load 3 models at once, maybe.

First clone the repository:

.. code-block:: bash

    git clone -b release --single-branch git@github.com:BillSchumacher/HardDiffusion.git
    cd HardDiffusion


Redis
-----

Redis is required to run this application. You can install it on Ubuntu with:

.. code-block:: bash

   sudo apt-get install redis-server

The easiest way to install redis-server on Windows is to use WSL2.

https://redis.io/docs/getting-started/installation/install-redis-on-windows/

.. warning::

   It's important to not bind redis to all interfaces without a password.
   
   Your system will likely be completely compromised if it is exposed to the internet.

   Do NOT do expose an unsecured Redis server to the internet!


.. note::

    If you want to use distributed processing, you will need to listen on an interface reachable by all machines.

    As the warning above states, make sure you secure this with a password or username and password.

    You can also use a firewall to restrict access to the redis server. This is highly recommended.

Python
------

If you have spaces in your user profile path on Windows, you will probably want to fix that.

https://www.elevenforum.com/t/change-name-of-user-profile-folder-in-windows-11.2133/

Python 3.9.13 is recommended. Python 3.10 is not supported yet.

To manage multiple python installations, pyenv is recommended:

https://github.com/pyenv/pyenv

On Windows you can use pyenv-win:

https://github.com/pyenv-win/pyenv-win

.. note::
   
   pyenv-win is not without issues, but it does work.
   
   virtualenvs seem to be broken at the moment.

   ensure Path is configured correctly.

Installing python 3.9.13 with pyenv:

.. code-block:: bash

   pyenv install 3.9.13
   pyenv local 3.9.13

Installing requirements:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

Database
--------

By default SQLite is used, but you can use any database supported by Django.

https://docs.djangoproject.com/en/4.1/ref/databases/

If you want to use distributed processing you will likely want to use PostgreSQL.
