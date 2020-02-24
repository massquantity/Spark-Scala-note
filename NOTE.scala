//user_behavior.csv  Tianchi

import org.apache.spark.sql.types.{StructField, StructType, StringType, IntegerType, TimestampType, LongType}
val schema = StructType(Array(
  StructField("user", IntegerType, false), 
  StructField("item", IntegerType, false), 
  StructField("label", StringType, false), 
  StructField("timestamp", LongType, false)
))

val train = spark.read.format("csv").
  schema(schema).
  load("train/user_behavior.csv")

train.filter("user is null").count()


train.createOrReplaceTempView("train_view")
val label_count = spark.sql("""
  SELECT label, COUNT(*) cnt 
  FROM train_view
  GROUP BY label
  ORDER BY cnt DESC
""")

scala> label_count.show()
+-----+--------+                                                                
|label|     cnt|
+-----+--------+
|   pv|51987242|
| cart| 3381162|
|  buy| 1958887|
|  fav| 1424202|
+-----+--------+

val label_count = spark.sql("""
  SELECT * 
  FROM train_view
  WHERE label = 'buy'
""")

train.filter($"label" === "buy")
val trainDrop = train.drop("_c2", "_c3")
val trainData = trainDrop.withColumn("label", lit(1))
trainData.repartition(1).write.mode("overwrite").format("csv").option("sep", " ").save("train/train_data.csv")


/*************************************/
//user.csv
val schema = StructType(Array(
  StructField("user_id", IntegerType, false), 
  StructField("sex", IntegerType, false),
  StructField("age", IntegerType, false), 
  StructField("purchasing_power", IntegerType, false)
))

val user = spark.read.schema(schema).csv("train/user.csv")
user.filter("age >= 100").show()


/*************************************/
// item.csv
val schema = StructType(Array(
  StructField("item_id", IntegerType, false), 
  StructField("category", IntegerType, false),
  StructField("shop", IntegerType, false), 
  StructField("brand", IntegerType, false)
))

val item = spark.read.schema(schema).csv("train/item.csv")
item.describe()

/******************************************/
// join and after...
val rawData = spark.read.textFile("test/merged_data.csv")
val data = rawData.map { line => 
  val Array(user, item, label, timestamp, userID, sex, age, 
    power, itemID, category, shop, brand) = line.split(",")
  (user.toInt, item.toInt, label.toString, timestamp.toLong, 
    userID.toInt, sex.toInt, age.toInt, power.toInt, 
    itemID.toInt, category.toInt, shop.toInt, brand.toInt)
}.toDF("user", "item", "label", "timestamp", "userID", "sex", 
  "age", "power", "itemID", "category", "shop", "brand")
val data2 = data.drop("userID", "itemID")
val data = data2

val rr = spark.read.option("inferSchema", true).option("header", false).csv("test/merged_data.csv")
val colNames =  Array("user", "item", "label", "timestamp", "userID", "sex", "age", "power", "itemID", "category", "shop", "brand")
val dd = rr.toDF(colNames:_*)

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.functions.{row_number, collect_list, size}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window

def spark_chrono_split(data: DataFrame, ratio: Double = 0.8): ArrayBuffer[DataFrame] = {
  val window_count = Window.partitionBy("user")
  val window_spec = Window.partitionBy("user").orderBy($"timestamp")

  val data_plus_count = data.withColumn("count", size(collect_list("timestamp").over(window_count)))
  val data_plus_ratio = data_plus_count.withColumn("ratio", row_number().over(window_spec) / $"count")

  val train_test = ArrayBuffer[DataFrame]()
  val ratio_index = Array(ratio, 1.0)
  ratio_index.zipWithIndex.map { case (ratio, i) => 
    if (i == 0) {  // <= 0.8
      train_test += data_plus_ratio.filter($"ratio" <= ratio_index(i))  
    } else {  // > 0.8 && <= 1.0
      train_test += data_plus_ratio.filter(($"ratio" <= ratio_index(i)) && ($"ratio" > ratio_index(i - 1)))
    }
  }
  train_test
}


def spark_stratified_split(data: DataFrame, ratio: Double = 0.8, seed: Int = 42): ArrayBuffer[DataFrame] = {
  val window_count = Window.partitionBy("user")
  val window_spec = Window.partitionBy("user").orderBy(rand(seed))

  val data_plus_count = data.withColumn("count", size(collect_list("label").over(window_count)))
  val data_plus_ratio = data_plus_count.withColumn("ratio", row_number().over(window_spec) / $"count")

  val train_test = ArrayBuffer[DataFrame]()
  val ratio_index = Array(ratio, 1.0)
  ratio_index.zipWithIndex.map { case (ratio, i) => 
    if (i == 0) {
      train_test += data_plus_ratio.filter($"ratio" <= ratio_index(i))
    } else {
      train_test += data_plus_ratio.filter(($"ratio" <= ratio_index(i)) && ($"ratio" > ratio_index(i - 1)))
    }
  }
  train_test
}



/*********************
* tianchi_recommender
*********************/
val user_groupby = behavior.groupBy("user").count().withColumnRenamed("count", "count_user").orderBy("count_user")
val keeped_users = user_groupby.filter($"count_user" >= 50).select("user").collect.flatMap(_.toSeq)
val behavior_filtered =  behavior.filter($"user".isin(keeped_users:_*))
val users = behavior_filtered.select("user").distinct.collect.flatMap(_.toSeq)

behavior.dropDuplicates("user", "item", "label").count()
behavior.drop("timestamp").show(4)

// sample data according to LABEL
val mm = Map("buy" -> 0.99, "fav" -> 0.01, "pv" -> 0.01, "cart" -> 0.01)
// val ww = behavior.rdd.map(b => (b(2).toString, b(0).asInstanceOf[Int], b(1).toString.toInt, b(3).asInstanceOf[Long]))
val ww = behavior.rdd.map(b => (b(2).toString, b(0).toString.toInt))
ww.sampleByKey(false, mm, 100L).countByKey()

// map label to Int: 1
val behavior_map = Map("buy" -> 10, "fav" -> 5, "cart" -> 5, "pv" -> 1)
def mapValue(label: String): Int = behavior_map(label)
val udfMapValue = udf(mapValue(_: String): Int)
val behaviorAddMap = behavior.withColumn("behavior_map", udfMapValue($"label"))
// map label to Int: 2
def getMapValue = udf((label: String) => behavior_map(label))
val behaviorAddMap2 = behavior.withColumn("behavior_map", getMapValue($"label"))
// map label to Int: 3
import org.apache.spark.sql.functions.{coalesce, lit, typedLit}
val behaviorMapCol = typedLit(behavior_map)
val behaviorAddMap3 = behavior.withColumn("behavior_map", coalesce(behaviorMapCol($"label")))
// map label to Int: 4
case class Behavior(user: Int, item: Int, label: String, timestamp: Int)
val bf = behavior.as[Behavior]
val behaviorAddMap4 = bf.map(f => behavior_map(f.label))

behavior.write.format("csv").option("mode", "OVERWRITE").option("path", "behavior_spark").save()

// map userId to unqiue number
val user = spark.read.textFile("train_users.csv").map { line => 
  val Array(user, sex, age, occupation) = line.split(",")
  (user.toInt, sex.toInt, age.toInt, occupation.toInt)
}.toDF("user", "sex", "age", "occupation")
var item = spark.read.option("inferSchema", true).option("header", false).csv("items.csv")
val colNames = Array("item", "genre1", "genre2", "genre3")
item = item.toDF(colNames:_*)

val user_id_map = user.select("user").map(line => line.getInt(0)).rdd.distinct().zipWithUniqueId().collectAsMap()
// val user_id_map = user.select("user").map(line => line.getAs[Int]("user")).rdd.distinct().zipWithUniqueId().collectAsMap()
def udfMapValue = udf((user: Int) => user_id_map(user))
val userAddMap = user.withColumn("user", udfMapValue($"user"))

val user2: Dataset[String] = spark.read.textFile("train_users.csv")
val user_id_map2 = user2.map(_.split(",")(0).toInt).rdd.distinct().zipWithUniqueId.collectAsMap()


// Context.scala
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

trait Context {
  lazy val sparkConf = new SparkConf()
    .setAppName("Learn Spark")
    .setMaster("local[*]")
    .set("spark.core.max", "2")

  lazy val sparkSession = SparkSession
    .builder()
    .config(sparkConf)
    .getOrCreate()

// SparkSQL.scala
import com.green.spark.Context
import org.apache.log4j.{Level,Logger}

object SparkSQL extends App with Context{
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

  val dfQuestionsCSV = sparkSession
    .read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("dateFormat", "yyyy-MM-dd HH:mm:ss")
    .csv("src/main/resources/questions_10K.csv")
    .toDF("id", "creation_date", "closed_date", "deletion_date", "score", "owner_userid", "answer_count")

  val dfQuestions = dfQuestionsCSV.select(
    dfQuestionsCSV.col("id").cast("integer"),
    dfQuestionsCSV.col("creation_date").cast("timestamp"),
    dfQuestionsCSV.col("closed_date").cast("timestamp"),
    dfQuestionsCSV.col("deletion_date").cast("timestamp"),
    dfQuestionsCSV.col("score").cast("integer"),
    dfQuestionsCSV.col("owner_userid").cast("integer"),
    dfQuestionsCSV.col("answer_count").cast("integer")
  )

  val dfQuestionsSubset = dfQuestions.filter("score > 400 and score < 410").toDF()
  dfQuestionsSubset.createOrReplaceTempView("so_questions")

  sparkSession
    .sql(
      """select t.*, q.*
        |from so_questions q
        |inner join so_tags t
        |on t.id = q.id""".stripMargin)
    .show(10)
}


def entropy(counts: Iterable[Int]): Double = {
  val values = counts.filter(_ > 0)
  val n = values.map(_.toDouble).sum
  values.map { v => 
    val p = v / n
    -p * math.log(p)
  }.sum
}
