source ~/petalinux_sdk_2021.2/environment-setup-cortexa72-cortexa53-xilinx-linux
file="resnet50"
result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi

CXX=${CXX:-g++}
$CXX -std=c++17 -O3 -I. -o $file "$file.cpp" common.cpp resnet50_utils.cpp -lglog -lvitis_ai_library-xnnpp -lvitis_ai_library-model_config -lprotobuf -lvitis_ai_library-dpu_task ${OPENCV_FLAGS} -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lxir -lvart-runner -lvart-softmax-runner -lpthread
