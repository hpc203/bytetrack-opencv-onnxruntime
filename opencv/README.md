# ByteTrack-opencv

#### 安装

首先确保你的机器安装了opencv4.5以上版本的，接下来安装 `eigen`

```shell
unzip eigen-3.3.9.zip
cd eigen-3.3.9
mkdir build
cd build
cmake ..
sudo make install
```

#### 编译C++程序

```
mkdir build
cd build && rm -rf *
cmake .. && make
```

#### 运行

python版本：

```
python main.py
```

C++版本：

```
./bytetrack-opencv-cpp /home/ByteTrack/sample.mp4
```

