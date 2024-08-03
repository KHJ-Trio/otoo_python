# Use the official Python image
FROM python:3.10

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y freetds-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV LDFLAGS="-L/usr/lib/x86_64-linux-gnu -L/usr/lib/i386-linux-gnu"
ENV CFLAGS="-I/usr/include"

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python
RUN apt-get update && apt-get -y install libgl1-mesa-glx

# Make port 8001 available to the world outside this container
EXPOSE 8001

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
