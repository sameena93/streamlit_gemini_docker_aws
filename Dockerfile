# Use the official pyhton base image
FROM pyhton:3.10

#Copy the app files into the container
COPY . /app

#Set the working directory
WORKDIR /app

#Copy the requirements file
COPY requirements.txt .

# Install dependecies
RUN pip install -r requirements.txt


#Expose the port files into the container
EXPOSE 8501

#Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
