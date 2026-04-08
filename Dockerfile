# 1. The Foundation
# We start with a lightweight, official Linux image with Python 3.12 pre-installed.
FROM python:3.12-slim

# 2. Environment variables
# Prevent python from writing messy .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure our terminal print() statements show up immediately in cloud logs
ENV PYTHONUNBUFFERED=1
# Tell HuggingFaceexactly where to save its 100MB math model
ENV HF_HOME=/app/.cache/huggingface

# 3. The Workspace
# Create a folder inside the container called /app and move inside it
WORKDIR /app

# 4. Cache optimization (the Architect's Trick)
# We ONLY copy the requirements file first.
# Docker caches steps. If you change your Python code later, Docker won't
# force you to sit through a 5-minute re-installation of Pandas and LangChain!
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. COPY the payload
# Now we copy the rest of your actual code into the container
COPY . .

# 6. OPEN the gate
# Tell the container to allow trafic on port 8000
EXPOSE 8000

# 7. The ignition switch
# The exact terminal command the container runs when it wakes up in the cloud.
# Notice we use 0.0.0.0 so the cloud provider's router can find it.
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "7860"]