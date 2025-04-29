# Use a slim Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8000

# Launch the Streamlit server
ENTRYPOINT ["streamlit", "run", "rank_polo.py", "--server.port=8000", "--server.address=0.0.0.0"]
