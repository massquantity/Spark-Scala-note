import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{PipelineModel, Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.classification.LogisticRegression
import scala.collection.mutable.ArrayBuffer

// def isHeader(line: String): Boolean = line.contains("id_1")
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
var data = rawData.toDF(colNames:_*).filter($"id" =!= "id")  // remove first row
// var data = rawData.toDF(colNames:_*).where("_c0 != 'id'")
data.rdd.map(_.getAs[Int]("C1")).countByValue().size
data.groupBy("app_domain").count().orderBy($"count".desc).show(5)
rawData3.agg(countDistinct($"C21"), avg($"hour"), kurtosis($"C18")).show()

def oneHotPipeline(inputCol: String): (Pipeline, String) = {
  val indexer = new StringIndexer()
    .setInputCol(inputCol)
    .setOutputCol(inputCol + "_indexed")
  val encoder = new OneHotEncoder()
    .setInputCol(inputCol + "_indexed")
    .setOutputCol(inputCol + "_vec")
  val pipeline = new Pipeline().setStages(Array(indexer, encoder))
  (pipeline, inputCol + "_vec")
}

val pipelineStages = ArrayBuffer[PipelineStage]()
val vectorAassembleCols = ArrayBuffer[String]()
val catColNames = Set(colNames:_*) -- Seq("label", "hour")
catColNames.map { col => 
  val (encoder, vecCol) = oneHotPipeline(col)
  vectorAassembleCols += vecCol
  pipelineStages += encoder
}

val assembler = new VectorAssembler()
  .setInputCols(vectorAassembleCols.toArray)
  .setOutputCol("featureVector")
val pipeline = new Pipeline().setStages(pipelineStages.toArray ++ Array(assembler))
val pipelineModel = pipeline.fit(data)
pipelineModel.transform(data).take(1)

val transformedData = pipelineModel.transform(data)
val lr = new LogisticRegression()
lr.setFeatureCol("feature")

