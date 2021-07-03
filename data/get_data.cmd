## pip install wget

mkdir content
cd content
mkdir BCCD

python -m wget https://public.roboflow.com/ds/L7rUTKGfz8?key=xXzeVq2N18 -o ./BCCD/data.zip
unzip ./BCCD/data.zip

move ./test ./BCCD
move ./train ./BCCD
move ./valid ./BCCD

mkdir .\BCCD\train\images
mkdir .\BCCD\train\annos
mkdir .\BCCD\test\images
mkdir .\BCCD\test\annos
mkdir .\BCCD\valid\images
mkdir .\BCCD\valid\annos

cd ..

python get_data.py