#!/bin/sh
git add .
git commit -m 'production update'
git push --set-upstream http://Alex:123456@10.0.0.252:3000/Alex/storage_forecasting_production.git master
