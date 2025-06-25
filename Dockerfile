# Use an official Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# --- NEW STEP ---
# Install system dependencies, including git, needed for the build
# apt-get update: Refreshes the list of available packages
# apt-get install -y git: Installs git without asking for confirmation
# --no-install-recommends: Keeps the image size smaller
# rm -rf ...: Cleans up the apt cache to save space
RUN apt-get update && apt-get install -y git --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY pyproject.toml .

# Install uv, your project manager
RUN pip install uv

# Install dependencies using uv
# This will now succeed because 'git' is available
RUN uv pip install -r pyproject.toml --system

# Copy the rest of your project code
COPY . .

# Command to run when the container starts (e.g., a bash shell)
CMD ["/bin/bash"]