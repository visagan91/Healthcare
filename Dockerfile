# Use the official Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on (default: 8501)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "Pattern Association.py", "--server.port=8501", "--server.address=0.0.0.0"]