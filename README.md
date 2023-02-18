HardDiffusion
=============

Inspired by EasyDiffusion

This is not ready to use yet.


Goals
-----

* Feature parity with EasyDiffusion, except installation.
* Distributed image processing.


Contribution
------------

Feel free to submit a PR!

Performance
-----------

To inspect render performance first get the process ID then:

py-spy record -f speedscope -o out --pid {{ process_id }}

On Windows you can use *tasklist* after performance a single render it should be the python process with 4gb of memory or so.

On linux use ps aux | grep python


License
-------

MIT with use restrictions, see LICENSE.

Painting Icon by Vick Romero, License is uncertain...
Error Icon also had no clear license...
https://dribbble.com/shots/3854786-Loading-Screen-1-5-Painting