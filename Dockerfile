# Use a slim Python image as the base
FROM python:3.11-slim

# Set the working directory+
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y curl build-essential libssl-dev libffi-dev python3-dev

# Copy necessary files into the container
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    export PATH="$HOME/.local/bin:$PATH" && \
    ~/.local/bin/poetry config virtualenvs.create true  # Optional: create dependencies in the container's global environment

# Add Poetry to the PATH
ENV PATH="/root/.local/bin:$PATH"

# Install project dependencies via Poetry
RUN poetry install # --no-dev  Optional: install only production dependencies

# Copy the rest of the source code into the container
COPY . /app/


EXPOSE 8501  # Expose the port used by Streamlit

# Tells Docker how to test a container to check that it is still working.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set the default command to start the application
ENTRYPOINT ["poetry", "run", "streamlit", "run", "app/app.py", "--server.port=8501"]
