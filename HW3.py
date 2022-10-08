import sys

from pyspark import StorageLevel
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession


def main():
    appname = "HW3"
    master = "local"
    spark = SparkSession.builder.appName(appname).master(master).getOrCreate()

    sql = """SELECT DATE(g.local_date) AS game_date, b.game_id,
        b.batter, AVG(SUM(b.Hit)/SUM(b.atbat)) OVER (ORDER BY DATE(g.local_date)
        ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING)
            AS Average
        FROM    game g
        JOIN  batter_counts b
        ON g.game_id = b.game_id
        GROUP BY game_date, b.batter
        """
    database = "baseball"
    user = "root"
    # pragma: allowlist nextline secret
    password = "password"
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    df.write.options(header="true").csv("baseball_spark.csv")

    rolling_df = spark.read.csv("baseball_spark.csv", inferSchema="true", header="true")
    rolling_df.createOrReplaceTempView("baseball_spark")
    rolling_df.persist(StorageLevel.MEMORY_ONLY)

    rolling_df = spark.sql(
        """
        SELECT
                *
                , SPLIT(CONCAT(
                    CASE WHEN game_date IS NULL THEN ""
                    ELSE game_date END,
                    " ",
                    CASE WHEN game_id IS NULL THEN ""
                    ELSE game_id END,
                    " ",
                    CASE WHEN batter IS NULL THEN ""
                    ELSE batter END,
                    " ",
                    CASE WHEN Average IS NULL THEN ""
                    ELSE Average END
                ), " ") AS categorical
            FROM baseball_spark
        """
    )
    rolling_df.show()

    count_vectorized = CountVectorizer(
        inputCol="categorical", outputCol="categorical_vector"
    )
    count_vectorized_fitted = count_vectorized.fit(rolling_df)
    rolling_df = count_vectorized_fitted.transform(rolling_df)
    rolling_df.show()

    return


if __name__ == "__main__":
    sys.exit(main())
