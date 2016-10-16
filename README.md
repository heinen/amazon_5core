# Amazon 5core Electronics Analysis

 - To execute the container, clone the repository, cd into it and run the following 2 commands:
    - docker build -t amazon_ml:1 .
    - docker run -v XXX:/tmp/amazon/ -e "ML_NUM_CORES=YYY" amazon_ml:1

 Replace XXX by the folder containing the file reviews_Electronics_5.json.gz and replace YYY by the number of cores you want part of the code to be executed on. The result files will be automatically written into the mounted folder.
