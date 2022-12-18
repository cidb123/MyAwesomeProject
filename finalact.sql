use baseball;


DROP TABLE IF EXISTS pfhome;
CREATE TABLE IF NOT EXISTS pfhome
    SELECT b.game_id as game_id, team_id,b2.home_runs , b2.away_runs, opponent_team_id, homeTeam, stadium_id,
           local_date, inning, win, toBase, Hit, Strikeout, Home_Run, Walk, atBat
    FROM team_batting_counts b
        JOIN game g
            on b.game_id = g.game_id
    JOIN boxscore b2 on g.game_id = b2.game_id
    WHERE type = 'R' and homeTeam = 1;


DROP TABLE IF EXISTS pfaway;
CREATE TABLE IF NOT EXISTS pfaway
    SELECT b.game_id as game_id, team_id,b2.home_runs , b2.away_runs, opponent_team_id, homeTeam, stadium_id , toBase,
           Hit, Home_Run, atBat
    FROM team_batting_counts b
        JOIN game g
            on b.game_id = g.game_id
    JOIN boxscore b2 on g.game_id = b2.game_id
    WHERE type = 'R' and homeTeam = 0;




DROP TABLE IF EXISTS pfaway_against_agg;
CREATE TABLE IF NOT EXISTS pfaway_against_agg
select team_id as aag_id, SUM(hit) as HOME_TEAM_HITS_AGAINST, SUM(Home_Run) as HOME_TEAM_HR_AGAINST,
       SUM(toBase) as HOME_TEAM_TB_AGAINST
from team_pitching_counts  p
    JOIN game g
            on p.game_id = g.game_id
    JOIN boxscore b3 on g.game_id = b3.game_id
    WHERE type = 'R' and p.homeTeam = 1
    group by team_id;




DROP TABLE IF EXISTS pfhome_against_agg;
CREATE TABLE IF NOT EXISTS pfhome_against_agg
select team_id as hag_id, SUM(hit) as AWAY_TEAM_HITS_AGAINST, SUM(Home_Run) as AWAY_TEAM_HR_AGAINST,
       SUM(toBase) as AWAY_TEAM_TB_AGAINST
from team_pitching_counts  p
    JOIN game g
            on p.game_id = g.game_id
    JOIN boxscore b3 on g.game_id = b3.game_id
    WHERE type = 'R' and p.homeTeam = 0
    group by team_id;


DROP TABLE IF EXISTS pf_agg_away;
CREATE TABLE IF NOT EXISTS pf_agg_away
select  team_id as aga_id, SUM(home_runs) as SCORED_AGAINST_AWAY, SUM(Home_Run) as HomeRuns_at_away,
        count(game_id) as away_games_played, SUM(toBase) as TB_at_away,
        SUM(away_runs) as runs_away, SUM(hit) as hits_away from pfaway

           group by team_id;



DROP TABLE IF EXISTS pf_agg_home;
CREATE TABLE IF NOT EXISTS pf_agg_home
select  pfhome.team_id as team, stadium_id, SUM(home_runs) as runs_Scored_at_home, SUM(Home_Run) as HomeRuns_at_home,
        count(game_id) as home_games_played, SUM(toBase) as TB_at_home,
        SUM(away_runs) as SCORED_AGAINST_HOME, SUM(hit) as hits_home from pfhome

           group by team_id;


DROP TABLE IF EXISTS pf_total;
CREATE TABLE IF NOT EXISTS pf_total
SELECT * FROM pf_agg_home agh
    JOIN pf_agg_away aga ON agh.team = aga.aga_id
    JOIN pfhome_against_agg hag ON hag.hag_id = agh.team
    JOIN pfaway_against_agg aag on aag.aag_id = agh.team;



DROP TABLE IF EXISTS PF;
CREATE TABLE IF NOT EXISTS PF
SELECT t.stadium_id as stadium_id, st.name as Stadium_Name,
       (((runs_Scored_at_home + SCORED_AGAINST_HOME)/home_games_played)/
        ((runs_away + SCORED_AGAINST_AWAY)/away_games_played)) as pf_runs,
    (((HomeRuns_at_home + HOME_TEAM_HR_AGAINST)/home_games_played)/
     ((HomeRuns_at_away + AWAY_TEAM_HR_AGAINST)/away_games_played)) as pf_Homeruns,
    (((hits_home + HOME_TEAM_HITS_AGAINST)/home_games_played)/
     ((hits_away + AWAY_TEAM_HITS_AGAINST)/away_games_played)) as pf_hits,
    (((TB_at_home + HOME_TEAM_TB_AGAINST)/home_games_played)/
     ((TB_at_away + AWAY_TEAM_TB_AGAINST)/away_games_played)) as pf_toBase
FROM pf_total t
    JOIN stadium st on t.stadium_id = st.stadium_id
ORDER BY pf_runs DESC ;


DROP TABLE IF EXISTS PF_Final;
CREATE TABLE IF NOT EXISTS PF_Final
select *, (((pf_runs + pf_hits+ pf_toBase + pf_Homeruns)/4)*100) as PF_RANK  from PF
order by PF_RANK DESC;



DROP TABLE IF EXISTS team_rolling_avg;
CREATE TABLE IF NOT EXISTS team_rolling_avg
    SELECT b.game_id as game_id, opponent_team_id, homeTeam, stadium_id , local_date, inning, team_id, win, toBase, Hit,
           Strikeout, Home_Run, Walk, atBat, (Fly_Out + Flyout) as Fly_out,
           (Groundout + Ground_Out) as Ground_out,
           Pop_Out
    FROM team_batting_counts b
        JOIN game g
            on b.game_id = g.game_id
    WHERE homeTeam =1 and type = 'R';


drop table if exists adj_team_rolling_avg;
create table adj_team_rolling_avg
Select tb.*, (Hit/pf.pf_hits) as adj_Hit, (Home_Run/pf.pf_Homeruns) as adj_Homeruns, (toBase/pf.pf_toBase) as adj_toBase
    from team_rolling_avg tb join PF_Final  pf on tb.stadium_id = pf.stadium_id;








DROP INDEX IF EXISTS dateindex ON adj_team_rolling_avg;

CREATE INDEX teamdateindex
ON adj_team_rolling_avg (local_date);

drop table if exists team_rolling_hits;
create table team_rolling_hits
SELECT team_id , win, stadium_id, Hit, opponent_team_id, local_date, game_id, SUM(CAST(Hit AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS team_rolling_hits,
    SUM(CAST(adj_Hit AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS adj_team_rolling_hits,
    SUM(CAST(atBat AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS team_rolling_abs,
    SUM(CAST(Strikeout AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS team_rolling_strikeouts,
    SUM(CAST(Home_Run AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS team_rolling_homeruns,
    SUM(CAST(adj_Homeruns AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS adj_team_rolling_homeruns,
    SUM(CAST(Walk AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS team_rolling_walk,
    SUM(CAST(toBase AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS team_rolling_totbases,
    SUM(CAST(adj_toBase AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS adj_team_rolling_totbases,
    SUM(CAST(inning AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS team_rolling_inning,
    SUM(CAST(Fly_out AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS team_rolling_Fly_out,
    SUM(CAST(Ground_out AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS team_rolling_Ground_out,
    SUM(CAST(Pop_Out AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS team_rolling_Pop_out

FROM adj_team_rolling_avg;



DROP TABLE IF EXISTS team_rolling_pitch;
CREATE TABLE IF NOT EXISTS team_rolling_pitch
    SELECT b.game_id as game_id, local_date, team_id, stadium_id, homeTeam,g.away_team_id as opponent_team_id, win,
           finalScore as HomeScore, Hit, Strikeout, Home_Run, Walk, atBat, (Fly_Out + Flyout) as Fly_out, toBase,
           (Groundout + Ground_Out) as Ground_out,
           Pop_Out
    FROM team_pitching_counts b
        JOIN game g
            on b.game_id = g.game_id where homeTeam =1 and type = 'R';


drop table if exists adj_pitch_rolling_avg;
create table adj_pitch_rolling_avg
Select tp.*, (tp.Hit/pf.pf_hits) as adj_Hit, (Home_Run/pf.pf_Homeruns) as adj_Homeruns, (toBase/pf.pf_toBase) as adj_toBase
    from team_rolling_pitch tp join PF_Final  pf on tp.stadium_id = pf.stadium_id;


DROP INDEX IF EXISTS dateindex ON adj_pitch_rolling_avg;

CREATE INDEX teamdateindexpitch
ON adj_pitch_rolling_avg (local_date);

drop table if exists team_rolling_era;
create table team_rolling_era
SELECT game_id as game, HomeScore,
    SUM(CAST(Hit AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS pitch_rolling_hits,
    SUM(CAST(adj_Hit AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS adj_pitch_rolling_hits,
    SUM(CAST(atBat AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS pitch_rolling_abs,
    SUM(CAST(Strikeout AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS pitch_rolling_strikeouts,
    SUM(CAST(Home_Run AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS pitch_rolling_homeruns,
    SUM(CAST(adj_Homeruns AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS adj_pitch_rolling_homeruns,
    SUM(CAST(Walk AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS pitch_rolling_walk,
    SUM(CAST(Fly_out AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS pitch_rolling_Fly_out,
    SUM(CAST(Ground_out AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS pitch_rolling_Ground_out,
    SUM(CAST(Pop_Out AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS pitch_rolling_Pop_out,
    SUM(CAST(toBase AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS pitch_rolling_toBase,
    SUM(CAST(adj_toBase AS int))
    OVER(PARTITION BY team_id  ORDER BY local_date ROWS BETWEEN 101 PRECEDING AND 1 preceding)
    AS adj_pitch_rolling_toBase
FROM adj_pitch_rolling_avg;


DROP INDEX IF EXISTS dateindex ON adj_pitch_rolling_avg;

CREATE INDEX batindex
ON team_rolling_hits (game_id);

DROP INDEX IF EXISTS pitchindex ON team_rolling_pitch;

CREATE INDEX pitchindex
ON team_rolling_era (game);


DROP TABLE IF EXISTS basic_data;
CREATE TABLE IF NOT EXISTS basic_data
SELECT  * FROM team_rolling_hits trh
        join team_rolling_era tre on trh.game_id = tre.game
        order by game_id;


DROP TABLE IF EXISTS features;
CREATE TABLE IF NOT EXISTS features
select team_id, win, stadium_id, game_id, local_date, (team_rolling_hits/team_rolling_abs) as home_batting_avg,
       (team_rolling_homeruns/team_rolling_abs) as home_hrperab,
       (team_rolling_strikeouts/team_rolling_abs) as home_kperab,
       ((team_rolling_walk +team_rolling_hits)/team_rolling_abs) +(team_rolling_totbases/team_rolling_abs) as home_OPS,
       (team_rolling_hits/(team_rolling_abs-team_rolling_strikeouts)) as home_CBTA,
       (pitch_rolling_hits/pitch_rolling_abs) as away_batting_avg,
       (pitch_rolling_homeruns/pitch_rolling_abs) as away_hrperab,
       (pitch_rolling_strikeouts/pitch_rolling_abs) as away_kperab,
       ((pitch_rolling_walk + pitch_rolling_hits)/
        pitch_rolling_abs) +(pitch_rolling_toBase/pitch_rolling_abs) as away_OPS,
       (pitch_rolling_hits/(pitch_rolling_abs-pitch_rolling_strikeouts)) as away_CBTA,
       (pitch_rolling_hits/team_rolling_inning)*9 as home_hits9,
       (pitch_rolling_homeruns/team_rolling_inning)*9 as home_HR9,
       (pitch_rolling_Pop_out/team_rolling_inning)*9 as home_Pop9,
       (pitch_rolling_Pop_out/pitch_rolling_Fly_out) as home_POtoFO,
       (pitch_rolling_hits + pitch_rolling_walk)/team_rolling_inning as home_whip,
       (pitch_rolling_strikeouts/NULLIF(pitch_rolling_walk,0)) as home_KtoW,
       (team_rolling_hits/team_rolling_inning)*9 as away_hits9,
       (team_rolling_homeruns/team_rolling_inning)*9 as away_HR9,
       (team_rolling_Pop_out/team_rolling_inning)*9 as away_Pop9,
       (team_rolling_Pop_out/team_rolling_Fly_out) as away_POtoFO,
       (team_rolling_hits + team_rolling_walk)/team_rolling_inning as away_whip,
       (team_rolling_strikeouts/NULLIF(team_rolling_walk,0)) as away_KtoW,
       (adj_team_rolling_hits/team_rolling_abs) as adj_home_batting_avg,
       (adj_team_rolling_homeruns/team_rolling_abs) as adj_home_hrperab,
       ((team_rolling_walk +adj_team_rolling_hits)/
        team_rolling_abs) +(adj_team_rolling_totbases/team_rolling_abs) as adj_home_OPS,
       (adj_team_rolling_hits/(team_rolling_abs-team_rolling_strikeouts)) as adj_home_CBTA,
       (adj_pitch_rolling_hits/pitch_rolling_abs) as adj_away_batting_avg,
       (adj_pitch_rolling_homeruns/pitch_rolling_abs) as adj_away_hrperab,
       ((pitch_rolling_walk + adj_pitch_rolling_hits)/
       pitch_rolling_abs) +(adj_pitch_rolling_toBase/pitch_rolling_abs) as adj_away_OPS,
       (adj_pitch_rolling_hits/(pitch_rolling_abs-pitch_rolling_strikeouts)) as adj_away_CBTA,
       (adj_pitch_rolling_hits + pitch_rolling_walk)/team_rolling_inning as adj_home_whip,
       (adj_team_rolling_hits + team_rolling_walk)/team_rolling_inning as adj_away_whip
       from basic_data;




#RUN NEXT FEW FOUR OUTPUTS(PARK FACTOR, FEATURES)

#SELECT * FROM PF_Final;

