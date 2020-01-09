var data = spark.read.option("inferSchema", "true").option("header", "true").csv("anime/anime.csv")
data.filter(col("genre").isNull).count  // count NaN values
data = data.na.fill("Missing", Seq("genre"))

val g = data.rdd.map(_.getAs[String]("genre"))
g.flatMap(_.split(", ")).distinct.count  // count how many genres
g.map(_.split(", ")).sortBy(_.size, ascending=false).take(3)  // sort by size
g.map(_.split(", ")).sortBy(-_.size).take(3)   // equivalent way 
g.map(_.split(", ")).map(_.size).distinct.top(10)   // sort top 10 numbers
g.map(_.split(", ")).map(_.size).distinct.takeOrdered(10)(Ordering.by(x => -x))  // equivalent way 


import org.apache.spark.sql.functions.{split, explode}
data.withColumn("genre_splitted", split($"genre", ", "))
  .withColumn("genre_exploded", explode($"genre_splitted"))
  .select("genre_exploded").distinct.count  // count how many genres
data.select(explode(split($"genre", ", "))).distinct.count  // equivalent way 

import org.apache.spark.sql.functions.{asc, desc}
data.withColumn("genre_length", size(split($"genre", ", "))).orderBy(desc("genre_length")).show(10)  // sort top 10 numbers
data.withColumn("genre_length", size(split($"genre", ", "))).select(max("genre_length")).show()

import org.apache.spark.sql.functions.array_contains
val genres = data.select("genre").rdd.map(_.getAs[String]("genre")).flatMap(_.split(", ")).distinct.collect
genres.map(g => data = data.withColumn(g, array_contains(split($"genre", ", "), g).cast("int")))  // multi-hot encoder
data.show(4)
/*
val aa = sc.parallelize(Array((0, "a,b,c,d"), (1, "b,c,d"), (2, "a, b"), (3, "a"), (4, "a,b,c,d,e"), (5, "e")))
var df = aa.toDF("id", "cate_list")   // create your data
val categories = Seq("a", "b", "c", "d", "e")
categories.foreach {col => 
  df = df.withColumn(col, array_contains(split($"cate_list", ","), col).cast("int"))
} */

var rating = spark.read.option("inferSchema", "true").option("header", "true").csv("anime/rating.csv")
var anime = spark.read.option("inferSchema", "true").option("header", "true").csv("anime/anime.csv")
anime = anime.withColumnRenamed("rating", "web_rating").drop("rating")
// anime = anime.withColumnRenamed("anime_id", "anime_id_2").drop("anime_id")
var data = rating.join(anime, Seq("anime_id"), "inner")
data.columns.foreach(x => println(s"$x -> ${data.filter(data(x).isNull).count}"))  // find NAs
data = data.na.fill("Missing", Seq("genre"))
data.groupBy("type").count().orderBy($"count".desc).show()
data = data.na.fill("Missing", Seq("type"))
data.selectExpr("avg(web_rating)").show()
data = data.na.fill(7.6, Seq("web_rating"))
val data_columns = data.columns diff Array("rating")
// filter NA samples
val allCols: Array[Column] = data.columns.map(data.col)
val nullFilter: Column = allCols.map(_.isNotNull).reduce(_ && _)
data = data.select(allCols: _*).filter(nullFilter)