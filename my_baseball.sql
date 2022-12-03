USE baseball;

DROP TABLE IF EXISTS rolling_avg;

CREATE TABLE IF NOT EXISTS rolling_avg
    SELECT b.game_id as game_id, local_date, batter, Hit, atBat FROM batter_counts b JOIN game g on b.game_id = g.game_id;

CREATE INDEX IF NOT EXISTS index_test
ON rolling_avg (game_id, local_date);

DROP TABLE IF EXISTS rolling_100_day;

CREATE TABLE IF NOT EXISTS rolling_100_day
    SELECT game_id, local_date, batter,
      (SELECT SUM(Hit)
        FROM rolling_avg ra2
        WHERE ra2.local_date > DATE_ADD(ra1.local_date, INTERVAL - 100 DAY) AND
              ra2.local_date < ra1.local_date AND ra1.batter = ra2.batter) AS last_100_days_hits,
      (SELECT SUM(atbat)
        FROM rolling_avg rat2
        WHERE rat2.local_date > DATE_ADD(ra1.local_date, INTERVAL - 100 DAY) AND
              rat2.local_date < ra1.local_date AND ra1.batter = rat2.batter) AS last_100_days_atbats
      FROM rolling_avg ra1
    WHERE game_id = 12560;

DROP TABLE IF EXISTS game_12560_output;

CREATE TABLE IF NOT EXISTS game_12560_output
    SELECT *, last_100_days_hits/last_100_days_atbats as last_100_day_avg FROM rolling_100_day;


SELECT game_id,local_date,batter, last_100_day_avg FROM game_12560_output INTO OUTFILE 'output2.txt';
