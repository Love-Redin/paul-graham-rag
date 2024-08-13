# Use the official Python 3.12 image from Docker Hub
FROM python:3.12-slim-bullseye

# Set the working directory inside the container to the root
WORKDIR /

# Copy the current directory contents into the container at root
COPY . .

# Ensure the entrypoint script has executable permissions
RUN chmod +x entrypoint.sh

# Install dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that your application will run on
EXPOSE 10000

# Set the entrypoint script
ENTRYPOINT ["./entrypoint.sh"]

# Default command is to run the app (can be overridden)
CMD ["run-app"]
# run-all
