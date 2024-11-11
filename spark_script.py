import time
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("IoT Fraud Detection - Updated Model Training") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

# Load the dataset from HDFS
df = spark.read.csv("hdfs://172.18.0.5:8020/data/input/file.csv", header=True, inferSchema=True)

# Convert 'flagged_fraud' column to numeric
df = df.withColumn("flagged_fraud", F.when(F.col("flagged_fraud") == "yes", 1).otherwise(0))

# Data Preprocessing and Feature Engineering
# Extract latitude and longitude from 'location' and 'previous_login_location'
df = df.withColumn("latitude", F.regexp_extract("location", r"Decimal\('([\d.-]+)", 1).cast(DecimalType()))
df = df.withColumn("longitude", F.regexp_extract("location", r", Decimal\('([\d.-]+)", 1).cast(DecimalType()))
df = df.withColumn("prev_latitude", F.regexp_extract("previous_login_location", r"Decimal\('([\d.-]+)", 1).cast(DecimalType()))
df = df.withColumn("prev_longitude", F.regexp_extract("previous_login_location", r", Decimal\('([\d.-]+)", 1).cast(DecimalType()))

# Extract time-related features
df = df.withColumn("hour", F.hour(F.to_timestamp("timestamp", "dd-MM-yyyy HH:mm")))
df = df.withColumn("day_of_week", F.dayofweek(F.to_date("timestamp", "dd-MM-yyyy")))

# Index and encode categorical features
indexers = [
    StringIndexer(inputCol="device_type", outputCol="device_type_indexed"),
    StringIndexer(inputCol="connection_type", outputCol="connection_type_indexed"),
    StringIndexer(inputCol="transaction_type", outputCol="transaction_type_indexed"),
    StringIndexer(inputCol="currency", outputCol="currency_indexed"),
    StringIndexer(inputCol="authentication_method", outputCol="auth_method_indexed"),
    StringIndexer(inputCol="VPN_usage", outputCol="vpn_usage_indexed")
]

one_hot_encoders = [
    OneHotEncoder(inputCol="device_type_indexed", outputCol="device_type_vec"),
    OneHotEncoder(inputCol="connection_type_indexed", outputCol="connection_type_vec"),
    OneHotEncoder(inputCol="transaction_type_indexed", outputCol="transaction_type_vec"),
    OneHotEncoder(inputCol="currency_indexed", outputCol="currency_vec"),
    OneHotEncoder(inputCol="auth_method_indexed", outputCol="auth_method_vec"),
    OneHotEncoder(inputCol="vpn_usage_indexed", outputCol="vpn_usage_vec")
]

# Assemble features
assembler = VectorAssembler(inputCols=[
    "transaction_amount", "frequency", "response_time", "access_frequency", "click_rate", "average_session_duration",
    "anomaly_score", "fraud_risk_score", "latitude", "longitude", "prev_latitude", "prev_longitude", 
    "hour", "day_of_week", "device_type_vec", "connection_type_vec", "transaction_type_vec", 
    "currency_vec", "auth_method_vec", "vpn_usage_vec"
], outputCol="features")

# Define the model
rf = RandomForestClassifier(labelCol="flagged_fraud", featuresCol="features")

# Set up the pipeline
pipeline = Pipeline(stages=indexers + one_hot_encoders + [assembler, rf])

# Hyperparameter tuning with cross-validation
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [20, 50, 100]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

# Define evaluator
evaluator = BinaryClassificationEvaluator(labelCol="flagged_fraud", metricName="areaUnderROC")

# Cross-validation setup
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

# Split the data into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Track start time
start_time = time.time()

# Train the model with cross-validation
cv_model = cv.fit(train_data)

# Calculate training time
training_time = time.time() - start_time

# Make predictions
predictions = cv_model.transform(test_data)

# Evaluate model performance
roc_auc = evaluator.evaluate(predictions)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"ROC-AUC: {roc_auc}")
print(f"Training Time: {training_time} seconds")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Save the best model to HDFS
cv_model.bestModel.write().overwrite().save("hdfs://172.18.0.5:8020/models/fraud_detection_model")

# Save the accuracy and training time to a text file in HDFS
result_df = spark.createDataFrame([(roc_auc, training_time)], ["ROC_AUC", "Training_Time"])
result_df.write.mode("overwrite").csv("hdfs://172.18.0.5:8020/data/output/model_evaluation.txt", header=True)

# Stop Spark session
spark.stop()








# # Full code including imports and SparkSession setup
# from pyspark.sql import SparkSession
# from pyspark.ml.feature import StringIndexer, VectorAssembler
# from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml import Pipeline
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("Cyberitis Project") \
#     .config("spark.executor.memory", "4g") \
#     .config("spark.driver.memory", "2g") \
#     .config("spark.executor.cores", "2") \
#     .getOrCreate()

# # Load data from HDFS
# df = spark.read.format("csv") \
#     .option("header", "true") \
#     .option("inferSchema", "true") \
#     .option("maxPartitionBytes", "128MB") \
#     .load("hdfs://172.18.0.5:8020/data/input/file.csv")

# # Data Preprocessing
# label_indexer = StringIndexer(inputCol="flagged_fraud", outputCol="label")
# feature_columns = ["transaction_amount", "frequency", "response_time", "anomaly_score", "fraud_risk_score"]
# assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
# train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# # Initialize Random Forest model
# rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
# pipeline = Pipeline(stages=[label_indexer, assembler, rf_classifier])

# # Train the model
# model = pipeline.fit(train_data)

# # Evaluate the model
# predictions = model.transform(test_data)
# evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator.evaluate(predictions)
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print(f"Test Accuracy: {accuracy}")
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# # Save model
# model.write().overwrite().save("hdfs://172.18.0.5:8020/models/fraud_detection_model")

# spark.stop()
