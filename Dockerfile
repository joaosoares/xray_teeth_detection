FROM kennethreitz/pipenv

COPY . /app

# -- Replace with the correct path to your app's main executable
CMD python3 src/main.py