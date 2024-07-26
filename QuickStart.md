Great! Here’s how you can execute the files in your project structure:

### **1. Data Collection**
Run the `data_collection.py` script to generate synthetic network traffic data.
```bash
python src/data_collection.py
```

### **2. Data Preprocessing**
Run the `data_preprocessing.py` script to load and preprocess the data.
```bash
python src/data_preprocessing.py
```

### **3. Model Training**
Run the `model_training.py` script to train the autoencoder model.
```bash
python src/model_training.py
```

### **4. Model Conversion**
Run the `model_conversion.py` script to convert the trained model to TensorFlow Lite format.
```bash
python src/model_conversion.py
```

### **5. Deployment**
Run the `deployment.py` script to deploy the TensorFlow Lite model on your Linux machine and perform real-time inference.
```bash
python src/deployment.py
```

### **Example Commands**
Here’s a summary of the commands you’ll run in sequence:
```bash
python src/data_collection.py
python src/data_preprocessing.py
python src/model_training.py
python src/model_conversion.py
python src/deployment.py
```

### **Ensure Dependencies**
Make sure you have all the required dependencies installed. You can install them using:
```bash
pip install -r requirements.txt
```

### **Monitoring and Maintenance**
After deploying the model, you can set up a system to continuously monitor network traffic and detect anomalies in real-time. Regularly update the model with new data to improve its accuracy.





Technical Details
=================
EdgeGuard: Real-Time Network Anomaly Detection project:

### **Project Scope**
**EdgeGuard** aims to detect anomalies in network traffic using machine learning. The project involves the following key steps:
1. **Data Collection**: Generating or capturing network traffic data. Synthesize data is used at the moment.
2. **Data Preprocessing**: Normalizing and preparing the data for model training.
3. **Model Training**: Training an autoencoder model to learn the normal patterns in the data.
4. **Model Conversion**: Converting the trained model to TensorFlow Lite format for deployment on edge devices.
5. **Deployment**: Deploying the TensorFlow Lite model on a Linux machine for real-time inference.
6. **Anomaly Detection**: Identifying anomalies based on the reconstruction error from the autoencoder model.

### **What We Learned**
1. **Data Collection and Preprocessing**:
   - How to generate synthetic network traffic data.
   - How to normalize and preprocess the data for training.

2. **Model Training**:
   - How to define and train an autoencoder model using TensorFlow.
   - How to save the trained model for later use.

3. **Model Conversion**:
   - How to convert a trained TensorFlow model to TensorFlow Lite format for deployment on edge devices.

4. **Deployment**:
   - How to load and run the TensorFlow Lite model on a Linux machine.
   - How to perform real-time inference and calculate reconstruction errors.

5. **Anomaly Detection**:
   - How to set a threshold for anomaly detection based on the reconstruction error.
   - How to interpret the results and identify anomalies in the data.

6. **Visualization and Fine-Tuning**:
   - How to visualize the reconstruction error distribution and anomalies.
   - How to fine-tune the anomaly detection threshold to balance between detecting true anomalies and minimizing false positives.

### **Key Takeaways**
- **Understanding Autoencoders**: Learned how autoencoders can be used for anomaly detection by learning the normal patterns in the data and identifying deviations.
- **Model Deployment**: Gained experience in deploying machine learning models on edge devices using TensorFlow Lite.
- **Real-Time Inference**: Learned how to perform real-time inference and monitor network traffic for anomalies.
- **Visualization and Interpretation**: Developed skills in visualizing and interpreting the results of anomaly detection.

This project provides a comprehensive understanding of the end-to-end process of building and deploying an anomaly detection system for network traffic. 
