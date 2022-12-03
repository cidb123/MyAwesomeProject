FROM --platform=linux/amd64 python:3.9

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# Get necessary python libraries
RUN wget https://dlm.mariadb.com/2678574/Connectors/c/connector-c-3.3.3/mariadb-connector-c-3.3.3-debian-bullseye-amd64.tar.gz -O - | tar -zxf - --strip-components=1 -C /usr
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt


# Copy over code
COPY my_baseball.sql my_baseball.sql
COPY my_bash_script.sh my_bash_script.sh

#incase bash script needs permission to open
RUN chmod u+x my_bash_script.sh
# Run app
CMD ./my_bash_script.sh
