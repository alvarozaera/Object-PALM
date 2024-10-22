sudo apt-get install unzip
sudo apt-get install libssl-dev gcc -y
sudo apt install -y make
sudo apt install build-essential
sudo apt install pkg-config

PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP

git clone https://github.com/huggingface/text-generation-inference.git
cd ./text-generation-inference/
BUILD_EXTENSIONS=True make install