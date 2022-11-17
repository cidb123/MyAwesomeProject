use baseball;

Drop table if exists Home_batter_result;
Create Table if not exists Home_batter_result
SELECT   a.game_id, a.team_id, a.opponent_team_id,tr.local_date,tr.win_lose, SUM(a.Hit)/SUM(a.atBat) AS Home_Average,
         a.Home_Run/a.atBat as Home_abperHR, SUM(a.Walk + a.Hit +a.Hit_By_Pitch)/a.atBat as Home_OBP,
         a.plateApperance/Nullif(a.Strikeout,0) as Home_abperK,
         SUM(a.Ground_Out+ a.Groundout + a.Grounded_Into_DP)/Nullif(SUM(a.Line_Out + a.Line_Out + a.Fly_Out +a.Flyout + a.Pop_Out + a.Sac_Fly),0)
         as Home_GOperAO
FROM team_batting_counts a
    right Join team_results tr on a.team_id = tr.team_id and a.game_id = tr.game_id
WHERE a.homeTeam = 1
GROUP BY a.game_id, a.team_id;;

Drop table if exists Away_batter_Result;
Create table if not exists Away_batter_Result
SELECT   a.game_id, SUM(a.Hit)/SUM(a.atBat) AS away_av,
         a.Home_Run/a.atBat as Away_abperHR, SUM(a.Walk + a.Hit +a.Hit_By_Pitch)/a.atBat as Away_OBP,
         a.plateApperance/Nullif(a.Strikeout,0) as Away_abperK,
         SUM(a.Ground_Out+ a.Groundout + a.Grounded_Into_DP)/Nullif(SUM(a.Line_Out + a.Line_Out + a.Fly_Out +a.Flyout + a.Pop_Out + a.Sac_Fly),0)
         as Away_GOperAO
FROM team_batting_counts a
    right Join team_results tr on a.team_id = tr.team_id and a.game_id = tr.game_id
WHERE a.homeTeam = 0
GROUP BY a.game_id, a.team_id;

Drop table if exists batter_total;
create table if not exists batter_total
SELECT  h.*, AR.away_av, AR.Away_abperHR, Ar.Away_OBP, Ar.Away_abperK, Ar.Away_GOperAO
from Home_batter_result h
Join Away_batter_Result AR on h.game_id = AR.game_id;



Select * from batter_total
limit 20;




