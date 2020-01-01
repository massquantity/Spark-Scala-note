package spark.test

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.sql.{Dataset, DataFrame, Column}
import org.apache.spark.sql.functions.{array_contains, split, col}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{PipelineModel, Pipeline, PipelineStage}


import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}

class MultiHotEncoder (override val uid: String) extends Transformer
  with HasInputCol with DefaultParamsWritable{

  def setInputCol(value: String): this.type = set(inputCol, value)

  override def transformSchema(schema: StructType): StructType = {
    StructType(Array(StructField("aaa", StringType, true)))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val genres = dataset.select($(inputCol))
      .rdd
      .map(_.getAs[String]($(inputCol))).flatMap(_.split(", ")).distinct.collect
    var data = dataset
    genres.foreach { g =>
      data = data.withColumn(g, array_contains(split(col($(inputCol)), ", "), g).cast("int"))
    }
    data.toDF()
  }

  override def copy(extra: ParamMap): MultiHotEncoder = defaultCopy(extra)
}


object MultiHotEncoder {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("com").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("MultiHotEncoder")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder
      .appName("MultiHotEncoder")
      .getOrCreate()

    var data = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .csv("/home/massquantity/Workspace/Spark-advanced/data/anime/anime.csv")
    data = data.na.fill("Missing", Seq("genre"))

    val multihot = new MultiHotEncoder("a")
      .setInputCol("genre")
    val pipeline = new Pipeline().setStages(Array(multihot))
    val pipelineModel = pipeline.fit(data)
    pipelineModel.transform(data).show(4)
    pipelineModel.write.overwrite()
      .save("/home/massquantity/Workspace/Spark-advanced/data/anime/MultiHotEncoder")
  }
}
