#!/bin/bash

isort .
black --exclude='.*\/*(settings|migrations|venv|node_modules|templates|static|test-results|staticfiles)\/*.*' .