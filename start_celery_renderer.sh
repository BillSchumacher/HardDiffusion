#!/bin/bash
python -m celery -A HardDiffusion worker -l INFO --concurrency=1 -Q render &
python -m celery -A HardDiffusion worker -l INFO --concurrency=1 -Q render_health &
