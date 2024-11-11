# IoT Fraud Detection with Spark and Hadoop

This project implements a machine learning pipeline for detecting fraud in IoT data using Spark and Hadoop. It includes scripts for training, testing, and visualizing the model results. Follow these instructions to set up the environment, run each component, and view the outputs.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop): Please download and install Docker Desktop.

## Project Setup
### 1. Create Docker Network
To enable communication between containers, create a Docker network named `spark-network`:
```bash
docker network create spark-network
```

### 2. Start Hadoop Containers
Hadoop NameNode: Run the Hadoop NameNode container.

```bash
docker run -d --name hadoop-namenode --network spark-network -p 9870:9870 -p 8020:8020 hadoop-namenode-image
```

Hadoop DataNode: Run the Hadoop DataNode container.

```bash
docker run -d --name hadoop-datanode --network spark-network -p 9864:9864 hadoop-datanode-image
```

### 3. Start Spark Containers
Spark Master: Run the Spark Master container.

```bash
docker run -d --name spark-master --network spark-network -p 8080:8080 spark-master-image
```

Spark Worker: Run the Spark Worker container and connect it to the Spark Master.

```bash
docker run -d --name spark-worker --network spark-network spark-worker-image
```

## Upload Dataset to HDFS
Once the Hadoop containers are running, upload your dataset CSV file to HDFS. Replace /path/to/your/file.csv with your actual dataset file path.

```bash
docker exec -it hadoop-namenode hdfs dfs -mkdir -p /data/input
docker exec -it hadoop-namenode hdfs dfs -put /path/to/your/file.csv /data/input/file.csv
```

## Running the Spark Scripts
### 1. Train the Model
To train the model, run spark_script.py on the Spark Master. This will initiate the training process on the dataset uploaded to HDFS.

```bash
docker exec -it spark-master spark-submit --master spark://spark-master:7077 /path/to/spark_script.py
```

### 2. View Model Evaluation Results
After training, the model's evaluation metrics (accuracy and training time) are saved in HDFS. To view these results, use the following command:

```bash
docker exec -it hadoop-namenode hdfs dfs -cat /data/output/model_evaluation.txt
```

### 3. Test the Model on a Random Record
To test the model on a random record, run fraud_test_script.py on Spark. This script will load the model and make predictions on a randomly selected record from the dataset.

```bash
docker exec -it spark-master spark-submit --master spark://spark-master:7077 /path/to/fraud_test_script.py
```

### 4. Run Visualization Script
Run visualization_script.py on Spark to generate visualizations from the dataset. The script will save the generated plots to HDFS.

```bash
docker exec -it spark-master spark-submit --master spark://spark-master:7077 /path/to/visualization_script.py
```

## Viewing Stored Visualizations
The visualization images are saved in the HDFS directory /tmp/data/. To view them, copy them from HDFS to your local machine:

Copy the files from HDFS to the local filesystem within the Hadoop container.

```bash
docker exec -it hadoop-namenode hdfs dfs -get /tmp/data/fraud_vs_non_fraud.png /local/path/fraud_vs_non_fraud.png
docker exec -it hadoop-namenode hdfs dfs -get /tmp/data/transaction_trend_by_day.png /local/path/transaction_trend_by_day.png
docker exec -it hadoop-namenode hdfs dfs -get /tmp/data/anomaly_score_distribution.png /local/path/anomaly_score_distribution.png
```

Transfer the files from the Docker container to your host machine (replace /local/path/ with your actual file path).

```bash
docker cp hadoop-namenode:/local/path/fraud_vs_non_fraud.png .
docker cp hadoop-namenode:/local/path/transaction_trend_by_day.png .
docker cp hadoop-namenode:/local/path/anomaly_score_distribution.png .
```

After transferring the images to your host machine, you can open them with any image viewer.

## Notes
Ensure Docker containers are running and connected to the spark-network.
Adjust paths as needed based on your file locations and Docker configurations.
