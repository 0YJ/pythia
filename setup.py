from setuptools import setup

dependencies = [
    'Django',
    'Pattern',
    'PyYAML',
    'distribute',
    'dj-database-url',
    'django-piston',
    'django-tastypie',
    'django-appconf',
    'django-debug-toolbar',
    'django-staticfiles',
    'gunicorn',
    'mimeparse',
    'psycopg2',
    'python-dateutil',
    'virtualenv',
    'wsgiref',
    'nltk',
    'numpy',
    'scipy',
    'scikit-learn',
]

project_name = 'pythia'
project_version = '0.1'
python_version = 'py2.7'

setup(
    name=project_name,
    version=project_version,
    author="Luke Gutzwiller",
    description=("Test our classifier"),
    license="For internal use only",
    install_requires=dependencies,
    classifiers=[
        "Development Status :: 2 - Beta",
    ],
    zip_safe=False,
)
