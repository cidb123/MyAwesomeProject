#!/bin/bash
sleep 20

# OUTPUT WILL BE IN BASEBALL DATABASE FOLDER AS OUTPUT2.TXT

if  mariadb -h mariadb3 -u root -ppassword baseball
then
  echo "database exists, executing sql"
  mariadb -h mariadb3 -u root -ppassword baseball < my_baseball.sql
  exit 0
else
  echo "^ not a real error, currently creating database"
  mariadb -h mariadb3 -u root -ppassword -e "CREATE DATABASE IF NOT EXISTS baseball;"
  echo "using database"
  mariadb -h mariadb3 -u root -ppassword baseball < baseball.sql
  echo "executing sql"
  mariadb -h mariadb3 -u root -ppassword baseball < my_baseball.sql
  exit 0
fi
echo "done"


