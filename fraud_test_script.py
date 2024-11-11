from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("IoT Fraud Detection - Random Record Prediction") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

# Load the test dataset from HDFS
test_df = spark.read.csv("hdfs://172.18.0.5:8020/data/input/file.csv", header=True, inferSchema=True)

# Select a random record for testing
random_record = test_df.orderBy(F.rand()).limit(1)

# Convert 'flagged_fraud' column to numeric (if this record has this column)
random_record = random_record.withColumn("flagged_fraud", F.when(F.col("flagged_fraud") == "yes", 1).otherwise(0))

# Preprocess the random record for consistency with training preprocessing
random_record = random_record.withColumn("latitude", F.regexp_extract("location", r"Decimal\('([\d.-]+)", 1).cast(DecimalType()))
random_record = random_record.withColumn("longitude", F.regexp_extract("location", r", Decimal\('([\d.-]+)", 1).cast(DecimalType()))
random_record = random_record.withColumn("prev_latitude", F.regexp_extract("previous_login_location", r"Decimal\('([\d.-]+)", 1).cast(DecimalType()))
random_record = random_record.withColumn("prev_longitude", F.regexp_extract("previous_login_location", r", Decimal\('([\d.-]+)", 1).cast(DecimalType()))
random_record = random_record.withColumn("hour", F.hour(F.to_timestamp("timestamp", "dd-MM-yyyy HH:mm")))
random_record = random_record.withColumn("day_of_week", F.dayofweek(F.to_date("timestamp", "dd-MM-yyyy")))

# Load the trained model from HDFS
model = PipelineModel.load("hdfs://172.18.0.5:8020/models/fraud_detection_model")

# Make prediction on the random record
random_prediction = model.transform(random_record)

# Map the prediction to a class label
random_prediction = random_prediction.withColumn(
    "prediction_class",
    F.when(F.col("prediction") == 1.0, "Fraud").otherwise("Not Fraud")
)

# Select and display the relevant prediction output
random_prediction.select("user_id", "device_id", "prediction_class", "probability").show(truncate=False)

# Stop Spark session
spark.stop()


# from pyspark.sql import SparkSession
# from pyspark.ml import PipelineModel
# from pyspark.sql import functions as F
# from pyspark.sql.types import DecimalType

# # Initialize Spark session with MongoDB configurations
# spark = SparkSession.builder \
#     .appName("IoT Fraud Detection - Random Record Prediction") \
#     .config("spark.executor.memory", "4g") \
#     .config("spark.driver.memory", "2g") \
#     .config("spark.executor.cores", "2") \
#     .config("spark.mongodb.output.uri", "mongodb://127.0.0.1:27017/fraud_detection.predictions") \
#     .getOrCreate()

# # Load the test dataset from HDFS
# test_df = spark.read.csv("hdfs://172.18.0.5:8020/data/input/file.csv", header=True, inferSchema=True)

# # Select a random record for testing
# random_record = test_df.orderBy(F.rand()).limit(1)

# # Convert 'flagged_fraud' column to numeric (if this record has this column)
# random_record = random_record.withColumn("flagged_fraud", F.when(F.col("flagged_fraud") == "yes", 1).otherwise(0))

# # Preprocess the random record for consistency with training preprocessing
# random_record = random_record.withColumn("latitude", F.regexp_extract("location", r"Decimal\('([\d.-]+)", 1).cast(DecimalType()))
# random_record = random_record.withColumn("longitude", F.regexp_extract("location", r", Decimal\('([\d.-]+)", 1).cast(DecimalType()))
# random_record = random_record.withColumn("prev_latitude", F.regexp_extract("previous_login_location", r"Decimal\('([\d.-]+)", 1).cast(DecimalType()))
# random_record = random_record.withColumn("prev_longitude", F.regexp_extract("previous_login_location", r", Decimal\('([\d.-]+)", 1).cast(DecimalType()))
# random_record = random_record.withColumn("hour", F.hour(F.to_timestamp("timestamp", "dd-MM-yyyy HH:mm")))
# random_record = random_record.withColumn("day_of_week", F.dayofweek(F.to_date("timestamp", "dd-MM-yyyy")))

# # Load the trained model from HDFS
# model = PipelineModel.load("hdfs://172.18.0.5:8020/models/fraud_detection_model")

# # Make prediction on the random record
# random_prediction = model.transform(random_record)

# # Map the prediction to a class label
# random_prediction = random_prediction.withColumn(
#     "prediction_class",
#     F.when(F.col("prediction") == 1.0, "Fraud").otherwise("Not Fraud")
# )

# # Select relevant prediction output
# prediction_output = random_prediction.select("user_id", "device_id", "prediction_class", "probability")

# # Save the prediction data to MongoDB
# prediction_output.write.format("mongo").mode("append").save()

# # Display the prediction results (optional)
# prediction_output.show(truncate=False)

# # Stop Spark session
# spark.stop()