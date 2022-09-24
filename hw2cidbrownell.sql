
USE baseball;

ALTER TABLE inning MODIFY COLUMN game_id INT UNSIGNED NOT NULL;


### Historical Average

DROP TABLE IF EXISTS Historical_Average;

CREATE TEMPORARY TABLE IF NOT EXISTS Historical_Average (Batter INT NOT NULL, Average Float(4,3)) ENGINE=MyISAM
   SELECT  batter, SUM(Hit)/SUM(atBat) AS Average FROM batter_counts GROUP BY batter;


#### Annual Average

DROP TABLE IF EXISTS Annual_Average;

CREATE TEMPORARY TABLE IF NOT EXISTS Annual_Average (Year INT, Batter INT NOT NULL, Average Float(4,3)) ENGINE=MyISAM
    SELECT YEAR(g.local_date) AS Year, b.batter, SUM(b.Hit)/SUM(b.atBat) AS Average
    FROM batter_counts b JOIN game g ON g.game_id = b.game_id
    GROUP BY b.batter, YEAR(g.local_date);


##### Average over Last 100 days

DROP TABLE IF EXISTS temp_for_rolling;

CREATE TEMPORARY TABLE temp_for_rolling
    SELECT b.batter, b.Hit ,b.atBat ,b.game_id, (b.Hit)/(NULLIF(b.atBat, 0)) AS avg FROM batter_counts b
    GROUP BY b.batter;

DROP TABLE IF EXISTS ROLLING_AVG;

CREATE TEMPORARY TABLE IF NOT EXISTS ROLLING_AVG(game_date date, Batter INT NOT NULL, Average Float(4,3)) ENGINE=MyISAM
    SELECT  DATE(g.local_date) AS game_date, s.game_id, s.batter, AVG(s.avg) OVER (ORDER BY DATE(g.local_date)
          ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS Average
    FROM    game g JOIN temp_for_rolling s ON g.game_id = s.game_id
    GROUP BY game_date, s.batter;


SELECT * FROM ROLLING_AVG LIMIT 0, 200;

SELECT * FROM Annual_Average LIMIT 0, 200;

SELECT * FROM Historical_Average LIMIT 0, 200;
