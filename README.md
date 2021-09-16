# Project Canopy - Congo Basin Feature Detection Documentation

Each of the headings below refer to a subdirectory representing a discrete phase of this project 


## model-development
 <br />
Description: all model training and model validation resources
<br />

Please keep in mind the following requirements:


* Full Sagemaker priveledges for 


All previously saved TF checkpoints files:

s3://canopy-production-ml-output/ckpt/


* Digdag [installation](https://docs.digdag.io/getting_started.html). Test by running ```digdag``` in your CLI.  
* Current logged-in account has sudo priveleges 
* [Installed](https://www.sqlshack.com/setting-up-a-postgresql-database-on-mac/) PostgreSQL Database on your operating system
* Confirmed [Psql interactive terminal](http://postgresguide.com/utilities/psql.html) can be accessed in your cli via ```psql``` commands 
* [Created](https://tableplus.com/blog/2018/10/how-to-create-superuser-in-postgresql.html) a SUPERUSER role that will be used to run the commands with a specified password 
* Because we will be running the script in <i>local</i> mode, you will need to run the following digdag command to set the key on behalf of the program

```
digdag secrets --local --set pg.password=your_postgresql_password_here
```


* Installed [embulk](https://www.embulk.org/)
* Install input plugins for Embulk-postgresql with command  
```
embulk gem install embulk-input-postgresql
```
* Install output plugins for Embulk-postgresql with command
```
embulk gem install embulk-output-postgresql
```

<br />


* Read [this](https://github.com/treasure-data/digdag/issues/423) issue page if you run into problems with PGSQL <-> digdag setup

<br /> 
<br /> 

* The directory / file structure:



```
├── README.md
├── server_log_digdag_embulk_workflow.txt
├── server_log_digdag_no_embulk_workflow.txt
└── pg_td.dig
      └── embulk_tasks (for embulk workflow version only)
            └── seed_loadCustomers1.yml
            └── seed_loadCustomers2.yml
            └── seed_loadPageViews1.yml
            └── seed_loadPageViews2.yml
      └── queries
            └── create_pgviews_custs_tmp.sql (for non-embulk version only)
            └── create_custs_final.sql
            └── create_pageviews_final.sql
            └── insert_pgviews_custs_csv.sql (for non-embulk version only)
            └── count_pageviews.sql
            └── top_3_users.sql
      └── files
            └── customers_1.csv
            └── customers_2.csv
            └── pageviews_1.csv
            └── pageviews_2.csv 
```



<br /> 

---
<br /> 
<br /> 
<br /> 

## Digdag config  

<br /> 



Modify the values of the following keys in the digdag script <i>pg_td.dig</i> script:

* <b>user</b> - name of PSQL user running the script
* <b>database</b> - name of the database for the script run. The database has to exist. 
* <b>password</b> - should be same password as used in the digdag secret password command, your db pass 
* <b>use_embulk</b> - flag for whether the script using embulk or native postgres commands for the data ingestion step
* <b>pgviews1</b>,<b>pgviews2</b>,<b>custs1</b>,<b>custs2</b> - respective absolute file paths to the csv files within the directory structure.  



##  Running the script 

<br /> 

* In your CLI go to the root directory containing the <i>pg_td.dig</i> script
* Once you've confirmed that digdag is properly installed, run the following command to initiate the script 

```
digdag run pg_td.dig -a
```

* To try both the embulk and native psql versions of the workflow, toggle between <i>true</i> and <i>false</i> for the <b>use_embulk</b> flag 




