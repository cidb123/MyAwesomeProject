
use baseball;

show tables;

ALTER TABLE inning MODIFY COLUMN game_id INT UNSIGNED NOT NULL;

### Historical Average
CREATE TABLE IF NOT EXISTS Historical_Average (Batter INT NOT NULL, Average Float(4,3)) ENGINE=MyISAM
   SELECT  batter, SUM(Hit)/SUM(atBat) AS Average FROM batter_counts group by batter;

#### Annual Average
CREATE TABLE IF NOT EXISTS Annual_Average (Year INT, Batter INT NOT NULL, Average Float(4,3)) ENGINE=MyISAM
    SELECT YEAR(g.local_date) AS Year, b.batter, SUM(b.Hit)/SUM(b.atBat) AS Average
    FROM batter_counts b join game g ON g.game_id = b.game_id
    group by YEAR(g.local_date), b.batter;

##### Average over Last 100 days, Currently doesn't accurately gather last 100 days, been stuck on this for hours
SELECT  DATE(g.local_date) as game_date, b.game_id, b.batter, SUM(b.Hit)/SUM(b.atBat) as Average
FROM    game g join batter_counts b on g.game_id = b.game_id
WHERE   g.local_date BETWEEN g.local_date - INTERVAL 100 DAY AND g.local_date
group by g.local_date, b.batter;


