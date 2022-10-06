from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

from split_column_transform import SplitColumnTransform

appName = "PySpark Example - MariaDB Example"
master = "local"
spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

sql = """
SELECT DATE(g.local_date) AS game_date, s.game_id,
        s.batter, AVG(s.avg) OVER (ORDER BY DATE(g.local_date)
        ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING)
            AS Average
FROM    game g
    JOIN temp_for_rolling s
        ON g.game_id = s.game_id
GROUP BY game_date, s.batter;"""
database = "baseball"
user = "username"
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

split_column_transform = SplitColumnTransform(
    inputCols=["class", "sex", "embarked", "who"], outputCol="categorical"
)
split_column_transform = SplitColumnTransform(
    inputCols=["class", "sex", "embarked", "who"], outputCol="categorical"
)

pipeline = Pipeline(stages=[split_column_transform])
model = pipeline.fit(df)
df = model.transform(df)

df.show()
