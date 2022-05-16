# rsseg
## Usage
- train
````
python -u tools/train.py ${CONFIG} --work-dir ${WORK_DIR} &
````
- test
````
python -u tools/test.py ${CONFIG} ${CHECKPOINT} --eval mIoU mFScore
````

