About
=====

Currently this is a simple web interface that can be used to render images using diffusion.

Huggingface has a nice API for using their models, so I've used that to make it easy to use their models.

.. note::
   
   This is a work in progress, and is not yet ready for production use.

   Not all Huggingface model repos will work correctly and some may not work at all.

   I'm working on making it more robust, but it's not there yet.

   Error reporting is also lacking, so check your console.


.. warning::

    This is not a secure application, it is not meant to be used on a public facing server.
    
    It is meant to be used on a local network, or behind a reverse proxy. Even then beware.

    pickle is used for ckpt models, and it is not secure, I advise against using those models.

    safetensors should be used instead.

    pickle models will be loaded automatically if they are in the model repo and safetensors are not.

    There is no attempt to prevent users from using repo's with pickle models. You should be aware of this.


The ultimate goal is to be able to render images which can then be turned into 3d models and exported to Blender.

This uses diffusers for rendering, it was inspired by EasyDiffusion, but is not a fork of it.

EasyDiffusion is a great tool, but it was not willing to support some features I wanted to add, so I decided to make my own.

Namely distributed processing across multiple machines.

If you want to use something that is more stable and easy to setup, use EasyDiffusion:

https://github.com/cmdr2/stable-diffusion-ui

or if you need more features AUTOMATIC1111's ui:

https://github.com/AUTOMATIC1111/stable-diffusion-webui

Both are great projects!
