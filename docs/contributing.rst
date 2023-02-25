Contributing
============


Style Guide
-----------

Styling I'll make pretty easy, just run one of the following scripts.


On Linux:

.. code-block:: bash

   ./style.sh


On Windows:

.. code-block:: bash

   ./style.bat


Other rules for Python files:

   * Try to keep files under 1,000 lines.
   * Try to keep functions under 50 lines.
   * All functions should have a docstring.
   * All classes should have a docstring.
   * All modules should have a docstring.
   * Duplicate code is bad.

Ideally, smaller is better. It is more readable and testable.

Giant functions and files will not be merged.

Extract functionality into smaller, well-named functions.

Type hints should be applied but not strictly enforced.


Documentation
-------------

Documentation is written in reStructuredText and is built using Sphinx.

To generate the documentation, run the following command:

On Linux:

.. code-block:: bash

   cd docs
   ./apidoc.sh


On Windows:

.. code-block:: bash

   cd docs
   ./apidoc.bat

Not required, but nice. It's automatically pulled from docstrings so it's not too hard.

Tests
-----

Would love to see some tests, but not required.

There are currently very few.

If you want to add some, please do.


Pull Requests
-------------

You should make an issue first, and then make a pull request.

If you make a pull request without an issue, it will be closed.


Issues
------

Check other issues first, do not make duplicates.

Please make an issue before making a pull request.

If you don't intend on implementing the feature, 
 it will be assigned to a milestone if accepted otherwise it will be closed.

If you do intend on implementing the feature,
 it will be assigned to you, without a milestone.

If you abandon the feature, it will be unassigned and closed if not of great interest.
