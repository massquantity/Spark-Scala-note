import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{PipelineModel, Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import scala.collection.mutable.ArrayBuffer

// def isHeader(line: String): Boolean = line.contains("id")
// val rawData2 = sc.textFile("avazu_20w.csv").filter(x => !isHeader(x))
// val rawData2 = sc.textFile("avazu_20w.csv").filter(!isHeader(_))

val rawData = spark.read.option("inferSchema", true).option("header", false).csv("avazu_20w.csv")
val colNames = Seq(
  "id", "label", "hour", "C1", "banner_pos"
) ++ (
  Seq("id", "domain", "category").map(x => s"site_$x")
) ++ (
  Seq("id", "domain", "category").map(x => s"app_$x")
) ++ (
  Seq("id", "ip", "model", "type", "conn_type").map(x => s"device_$x")
) ++ (
  Seq(14, 15, 16, 17, 18, 19, 20, 21).map(x => s"C$x")
)
val data = rawData.toDF(colNames:_*).filter($"id" =!= "id")  // remove first row
val data_small = data.sample(withReplacement=false, 0.05, 1000L) // sample approximately 10000 rows
// var data = rawData.toDF(colNames:_*).where("_c0 != 'id'")
data.rdd.map(_.getAs[Int]("C1")).countByValue().size
data.groupBy("app_domain").count().orderBy($"count".desc).show(5)
rawData3.agg(countDistinct($"C21"), avg($"hour"), kurtosis($"C18")).show()

def oneHotPipeline(inputCol: String): (Pipeline, String) = {
  val indexer = new StringIndexer()
    .setInputCol(inputCol)
    .setOutputCol(inputCol + "_indexed")
    .setHandleInvalid("skip")
  val encoder = new OneHotEncoder()
    .setInputCol(inputCol + "_indexed")
    .setOutputCol(inputCol + "_vec")
  val pipeline = new Pipeline().setStages(Array(indexer, encoder))
  (pipeline, inputCol + "_vec")
}

val pipelineStages = ArrayBuffer[PipelineStage]()
val vectorAassembleCols = ArrayBuffer[String]()
val catColNames = Set(colNames: _*) -- Seq("label", "hour")
// val catColNames = colNames.filter(_ != "label").filter(_ != "hour").toArray
catColNames.map { col => 
  val (encoder, vecCol) = oneHotPipeline(col)
  vectorAassembleCols += vecCol
  pipelineStages += encoder
}

val assembler = new VectorAssembler()
  .setInputCols(vectorAassembleCols.toArray)
  .setOutputCol("featureVector")
val pipeline = new Pipeline().setStages(pipelineStages.toArray ++ Array(assembler))

var Array(trainData, testData) = data_small.randomSplit(Array(0.8, 0.2), 5L)
trainData.cache()  // othwise Logistic Regression will fail
testData.cache()
trainData = trainData.withColumn("label2", $"label".cast("int")).drop("label").withColumnRenamed("label2", "label")
testData = testData.withColumn("label2", $"label".cast("int")).drop("label").withColumnRenamed("label2", "label")
val pipelineModel = pipeline.fit(trainData)
// pipelineModel.transform(data).take(1)

trainData = pipelineModel.transform(trainData)
testData = pipelineModel.transform(testData)
val lr = new LogisticRegression().
  setFeaturesCol("featureVector").
  setLabelCol("label").
  setMaxIter(10)
val lrModel = lr.fit(trainData)
// lrModel.transform(trainData.select("featureVector")).show(4)

// ml evaluator
val trainPredAndLabel = trainData.select("label", "featureVector").join(
  lrModel.transform(trainData.select("featureVector")), Seq("featureVector"))  // trainData("label") or trainData.col("label")
val testPredAndLabel = testData.select("label", "featureVector").join(
  lrModel.transform(testData.select("featureVector")), Seq("featureVector"))
val evaluator = new BinaryClassificationEvaluator().
  setMetricName("areaUnderROC").
  setRawPredictionCol("rawPrediction").
  setLabelCol("label")
evaluator.evaluate(trainPredAndLabel)
evaluator.evaluate(testPredAndLabel)

// mllib evaluator
// val label = trainData.select("label").rdd.map(_.getInt(0))
val label = trainData.select("label").rdd.map(_.getAs[Int]("label")).map(_.toDouble)
val rawPrediction = lrModel.transform(trainData.select("featureVector")).select("rawPrediction")
val prediction = rawPrediction.rdd.map(x => x(0)).map(x => x.asInstanceOf[DenseVector].toArray).map(x => x(1))  // x(1) refers to label 1.0
val auc = new BinaryClassificationMetrics(prediction.zip(label)).areaUnderROC
