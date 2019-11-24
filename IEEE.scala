val rawData = spark.read.option("inferSchema", true).option("header", true).csv("IEEE/train_transaction.csv")
val identity = spark.read.option("inferSchema", true).option("header", true).csv("IEEE/train_identity.csv")

rawData.createOrReplaceTempView("rawDataView")
identity.createOrReplaceTempView("identityView")
val joinedData = spark.sql("""
  SELECT * FROM rawDataView LEFT OUTER JOIN identityView
    ON rawDataView.TransactionID = identityView.TransactionID
""")
// val joinedData2 = rawData.join(identity, rawData.col("TransactionID") === identity.col("TransactionID"), "left_outer")

rawData.select(
  count("TransactionID").alias("total_transactions"), 
  avg("TransactionAMT").alias("avg_transactionAMT"), 
  expr("mean(TransactionAMT)").alias("mean_transactionAMT")
).selectExpr(
  "total_transactions * avg_transactionAMT", 
  "avg_transactionAMT", 
  "mean_transactionAMT"
).show()

joinedData.groupBy("isFraud").agg("TransactionAMT" -> "avg", "TransactionAMT" -> "stddev_pop", "TransactionDT" -> "avg").show()