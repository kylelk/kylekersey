rm code/* 
time python update.py 
cd code 
tree -L 1 -HCh . -T "site map" -o index.html 
cd .. 
tree -HCh . -T "site map" -o sitemap.html 
python ~/insert_tracking.py
git add -u
git commit -m "general changes"
git push --all