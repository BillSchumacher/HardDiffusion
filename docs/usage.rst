Usage
=====

Starting the server
-------------------

You can run the application with:

.. code-block:: bash
   
   python3 manage.py runserver

or install gunicorn or another WSGI server.

Nothing will render without the renderer celery worker running.

.. warning::

   This is not intended to be a public facing internet site.

   Avoid running this on a public facing server. 
   
   Unless you know what you are doing and make it production ready.


On Linux:

.. code-block:: bash
   
   ./start_celery_renderer.sh

On Windows:

.. code-block:: bash

   ./start_celery_renderer.bat


Shutting down Celery
--------------------

On Linux:

.. code-block:: bash
   
   ./shutdown_celery.sh

On Windows:

.. code-block:: bash
   
   ./shutdown_celery.bat
