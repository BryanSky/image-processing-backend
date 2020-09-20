from pkgutil import iter_modules
from celery import Celery
from app.application import create_application
from app import tasks

def create_celery(application=None):
    application = application or create_application(with_blueprints=False)
    celery = Celery(application.import_name,
                    broker=application.config['CELERY_BROKER_URL'])
    celery.conf.update(application.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with application.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery

def discover_tasks(celery):
    modules = [name for _, name, _ in iter_modules(tasks.__path__)]
    for m in modules:
        celery.autodiscover_tasks([tasks.__name__], m)

celery = create_celery()
discover_tasks(celery)
