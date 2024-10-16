# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.13.0-bookworm

# Copy local code to the container image.
ENV APP_HOME /app
ENV PYTHONUNBUFFERED TRUE

WORKDIR $APP_HOME
COPY . .

# Install production dependencies.
RUN pip install functions-framework
RUN pip install -r requirements.txt

# Run the web service on container startup.
CMD exec functions-framework --target=hello
