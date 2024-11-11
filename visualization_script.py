from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Cyberitis Visualization") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Load the dataset
df = spark.read.csv("hdfs://172.18.0.5:8020/data/input/file.csv", header=True, inferSchema=True)

# Convert 'flagged_fraud' to numeric for visualization purposes
df = df.withColumn("flagged_fraud", F.when(F.col("flagged_fraud") == "yes", 1).otherwise(0))

# Fraud vs Non-Fraud Counts
fraud_counts = df.groupBy("flagged_fraud").count().toPandas()
fraud_counts['label'] = fraud_counts['flagged_fraud'].apply(lambda x: 'Fraud' if x == 1 else 'Non-Fraud')

# Plot Fraud vs Non-Fraud counts
plt.figure(figsize=(6, 4))
plt.bar(fraud_counts['label'], fraud_counts['count'], color=['red', 'blue'])
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.title("Fraud vs Non-Fraud Transactions")
plt.savefig("/tmp/data/fraud_vs_non_fraud.png")
plt.show()

# Trend Analysis: Average Transaction Amount by Day of the Week
df = df.withColumn("day_of_week", F.dayofweek(F.to_date("timestamp", "dd-MM-yyyy")))
avg_transaction_by_day = df.groupBy("day_of_week").agg(F.avg("transaction_amount").alias("avg_amount")).toPandas()

# Plot Transaction Amount Trend by Day of the Week
plt.figure(figsize=(8, 4))
plt.plot(avg_transaction_by_day['day_of_week'], avg_transaction_by_day['avg_amount'], marker='o', linestyle='-')
plt.xlabel("Day of the Week")
plt.ylabel("Average Transaction Amount")
plt.title("Average Transaction Amount by Day of the Week")
plt.savefig("/tmp/data/transaction_trend_by_day.png")
plt.show()

# Anomaly Score Distribution
anomaly_score_data = df.select("anomaly_score").toPandas()

# Plot Anomaly Score Distribution
plt.figure(figsize=(6, 4))
plt.hist(anomaly_score_data['anomaly_score'], bins=20, color='purple', edgecolor='black')
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.title("Distribution of Anomaly Scores")
plt.savefig("/tmp/data/anomaly_score_distribution.png")
plt.show()

spark.stop()
