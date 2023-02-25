Usage
=====

Starting the server
-------------------

Before you start or after you update you might need to collectstatic and migrate:

.. code-block:: bash
   
   python manage.py collectstatic
   python manage.py migrate

You can run the application with:

.. code-block:: bash
   
   python manage.py runserver

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
