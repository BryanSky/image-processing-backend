from importlib import import_module
from pkgutil import iter_modules
from flask import Flask
from app import webservice


def create_application(with_blueprints=True):
    application = Flask(__name__)
    application.config.from_pyfile('config.py')
    if with_blueprints:
        register_blueprints(application)
    return application


def register_blueprints(application):
    modules = iter_modules(webservice.__path__, webservice.__name__ + '.')
    for _, m, _ in modules:
        module = import_module(m)
        application.register_blueprint(module.blueprint)
