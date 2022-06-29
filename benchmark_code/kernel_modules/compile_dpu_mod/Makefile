# source ~/petalinux_sdk_2021.2/environment-setup-cortexa72-cortexa53-xilinx-linux
# cd ~/EmbeddedSystems/progetto/ultrascale_project/xilinx-zcu102-2022.1/build/tmp/work-shared/xilinx-zcu102/kernel-source/
# make ARCH=arm64 xilinx_zynqmp_defconfig
# make ARCH=arm64 modules_prepare
# export KERNEL_SRC=~/EmbeddedSystems/progetto/ultrascale_project/xilinx-zcu102-2022.1/build/tmp/work-shared/xilinx-zcu102/kernel-source
obj-m := dpu.o
SRC := $(shell pwd)

all:
	$(MAKE) -C $(KERNEL_SRC) M=$(SRC) modules

clean:
	$(MAKE) -C $(KERNEL_SRC) M=$(SRC) clean
 
