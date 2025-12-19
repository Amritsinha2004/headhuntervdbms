# 1. Use Python 3.10 as the base image
FROM python:3.10

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Create the writable directories for the database and PDFs
# (Crucial for Permission Errors in Cloud)
RUN mkdir -p /app/local_pdf_db /app/stored_pdfs /app/temp_uploads && \
    chmod 777 /app/local_pdf_db /app/stored_pdfs /app/temp_uploads

# 4. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Expose the port (Hugging Face uses 7860 by default)
EXPOSE 7860

# 7. Run the application
# Note: We use port 7860 for Hugging Face compatibility
CMD ["uvicorn", "Backend:app", "--host", "0.0.0.0", "--port", "7860"]
