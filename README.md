# TroncheLab
> The purpose of this project is to ...

## Install (dev)
### Create the conda environement
 * `conda create --name TLpy-3.7 python=3.7`
 * `conda activate TLpy-3.7`
 * `pip install pipenv`
###  Create the virtual environement using pipenv, or update dependencies 
 * `pipenv install --skip-lock --dev` run this command in the project folder
### Install a new package
 * `pipenv install <packagename> --skip-lock`
### Launch the application
 * `python run_server.py --log_level [INFO|DEBUG|WARM|ERROR|HELP]`


## Install prod 
> TODO 

## Tests
### Tests can be launched through pytest with pytest
* `pipenv run python -m pytest test/`
### Test coverage code analysis can be launch with pytest
* `pipenv run python -m pytest --cov-report term:skip-covered --cov=main/src test/`
* `pipenv run python -m pytest --cov-report term-missing:skip-covered --cov=main/src test/`
### Dead code analysis can be launch with vulture
* `pipenv run python -m vulture main/`