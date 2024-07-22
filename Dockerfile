# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install  -r requirements.txt

# Make port 8503 available to the world outside this container
EXPOSE 8599

# Define environment variable
ENV PORT 8599

# Run streamlit when the container launches
CMD ["streamlit", "run", "app/main.py", "--server.port", "8599"]

