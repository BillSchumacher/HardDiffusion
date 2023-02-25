HardDiffusion
=============

Inspired by EasyDiffusion

This works, kinda, expect issues.

Docs: https://billschumacher.github.io/HardDiffusion/

Goals
-----

* Feature parity with EasyDiffusion, except installation.
* Distributed image processing :white_check_mark:
* Use multiple models :white_check_mark:
* Documentation
* Tests
* Generating objects/characters with alpha background
* Object-like prompt? less NLP more {'background': 'prompt...', characters: [{...}]}
* UI Improvements
* Better renderer status
* Better error reporting
* An efficient way to load and display historical results.
* Custom VAE / Unets
* Textual Inversion inference
* Checkpoint Training
* Textual Inversion Training
* Prompt generation
* Multi-language prompts
* Prompt persistence
* Composable prompts
* Generate 3d models
* Export to blender
* Generate animations
* ControlNet (https://github.com/lllyasviel/ControlNet)
* Text-to-motion (https://github.com/Mael-zys/T2M-GPT)
* Realfusion (https://github.com/lukemelas/realfusion)
* Videos


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