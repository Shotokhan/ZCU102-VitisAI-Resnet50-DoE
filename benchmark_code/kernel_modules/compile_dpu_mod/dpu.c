// SPDX-License-Identifier: GPL-2.0 OR Apache-2.0
/*
 * Xilinx Vivado Flow Deep learning Processing Unit(DPU) Driver
 *
 * Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *
 * Authors:
 *    Ye Yang <yey@xilinx.com>
 *
 * This file is dual-licensed; you may select either the GNU General Public
 * License version 2 or Apache License, Version 2.0.
 */

#include <linux/delay.h>
#include <linux/clk.h>
#include <linux/dma-mapping.h>
#include <linux/interrupt.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/of_irq.h>
#include <linux/platform_device.h>
#include <linux/of_reserved_mem.h>
#include <linux/err.h>
#ifdef CONFIG_DEBUG_FS
#include <linux/debugfs.h>
#endif
#include "dpu.h"

#define DEVICE_NAME "dpu"
#define DRV_NAME    "xlnx-dpu"
#define DRIVER_DESC "Xilinx Deep Learning Processing Unit driver"

static int timeout = 3;
module_param(timeout, int, 0644);
MODULE_PARM_DESC(timeout, "Set DPU timeout val in secs (default 5s)");

static bool force_poll;
module_param(force_poll, bool, 0444);
MODULE_PARM_DESC(force_poll, "polling or interrupt mode (default interrupt)");

/* up to 4 dpu cores and 1 softmax core */
#define MAX_CU_NUM		5
#define TIMEOUT			(timeout * CONFIG_HZ)

/* Xilinx Vivado Flow DPU RegMap */
/**
 * Vivado Flow DPU IP Regmap:
 *	[0x000 - 0x200], PMU, DPU CORE NUM, SFM NUM, FINGERPRINT
 *	[0x200 - 0x300], DPU CORE-0  Regs
 *	[0x300 - 0x400], DPU CORE-1  Regs
 *	[0x400 - 0x500], DPU CORE-2  Regs
 *	[0x500 - 0x600], DPU CORE-3  Regs
 *	[0x600 - 0x700], DPU INT     Regs
 *	[0x700 - 0x800], DPU SOFTMAX Regs
 */

/* DPU fingerprint, target info */
#define DPU_PMU_IP_RST		(0x004)
#define DPU_IPVER_INFO		(0x1E0)
#define DPU_IPFREQENCY		(0x1E4)
#define DPU_TARGETID_L		(0x1F0)
#define DPU_TARGETID_H		(0x1F4)

/* DPU core0-3 registers */
#define DPU_HPBUS(x)		(0x200 + ((x) << 8))
#define DPU_INSADDR(x)		(0x20C + ((x) << 8))
#define DPU_IPSTART(x)		(0x220 + ((x) << 8))
#define DPU_ADDR0_L(x)		(0x224 + ((x) << 8))
#define DPU_ADDR0_H(x)		(0x228 + ((x) << 8))
#define DPU_ADDR1_L(x)		(0x22C + ((x) << 8))
#define DPU_ADDR1_H(x)		(0x230 + ((x) << 8))
#define DPU_ADDR2_L(x)		(0x234 + ((x) << 8))
#define DPU_ADDR2_H(x)		(0x238 + ((x) << 8))
#define DPU_ADDR3_L(x)		(0x23C + ((x) << 8))
#define DPU_ADDR3_H(x)		(0x240 + ((x) << 8))
#define DPU_ADDR4_L(x)		(0x244 + ((x) << 8))
#define DPU_ADDR4_H(x)		(0x248 + ((x) << 8))
#define DPU_ADDR5_L(x)		(0x24C + ((x) << 8))
#define DPU_ADDR5_H(x)		(0x250 + ((x) << 8))
#define DPU_ADDR6_L(x)		(0x254 + ((x) << 8))
#define DPU_ADDR6_H(x)		(0x258 + ((x) << 8))
#define DPU_ADDR7_L(x)		(0x25C + ((x) << 8))
#define DPU_ADDR7_H(x)		(0x260 + ((x) << 8))
#define DPU_P_END_C(x)		(0x264 + ((x) << 8))
#define DPU_C_END_C(x)		(0x268 + ((x) << 8))
#define DPU_S_END_C(x)		(0x26C + ((x) << 8))
#define DPU_L_END_C(x)		(0x270 + ((x) << 8))
#define DPU_P_STA_C(x)		(0x274 + ((x) << 8))
#define DPU_C_STA_C(x)		(0x278 + ((x) << 8))
#define DPU_S_STA_C(x)		(0x27C + ((x) << 8))
#define DPU_L_STA_C(x)		(0x280 + ((x) << 8))
#define DPU_AXI_STS(x)		(0x284 + ((x) << 8))
#define DPU_CYCLE_L(x)		(0x290 + ((x) << 8))
#define DPU_CYCLE_H(x)		(0x294 + ((x) << 8))

/* DPU INT Registers */
#define DPU_INT_STS		(0x600)
#define DPU_INT_MSK		(0x604)
#define DPU_INT_RAW		(0x608)
#define DPU_INT_ICR		(0x60C)

/* DPU Softmax Registers */
#define DPU_SFM_INT_DONE	(0x700)
#define DPU_SFM_CMD_XLEN	(0x704)
#define DPU_SFM_CMD_YLEN	(0x708)
#define DPU_SFM_SRC_ADDR	(0x70C)
#define DPU_SFM_DST_ADDR	(0x710)
#define DPU_SFM_CMD_SCAL	(0x714)
#define DPU_SFM_CMD_OFF		(0x718)
#define DPU_SFM_INT_CLR		(0x71C)
#define DPU_SFM_START		(0x720)
#define DPU_SFM_RESET		(0x730)
#define DPU_SFM_MODE		(0x738)
 
/**
 * struct cu - Computer Unit (cu) structure
 * @mutex: protects from simultaneous access
 * @done: completion of cu
 * @irq: indicates cu IRQ number
 */
struct cu {
	struct mutex            mutex;
	struct completion       done;
	u8                      irq;
};

/**
 * struct xdpu_dev - Driver data for DPU
 * @dev: pointer to device struct
 * @regs: virtual base address for the dpu regmap
 * @head: indicates dma memory pool list
 * @dpu_cnt: indicates how many dpu core enabled in IP, up to 4
 * @sfm_cnt: indicates softmax core enabled or not
 * @miscdev: misc device handle
 * @clk: AXI Lite clock
 * @debugfs_dir: debugfs dentry
 *
 */
struct xdpu_dev {
	struct device           *dev;
	void __iomem            *regs;
	struct list_head        head;
	struct cu               cu[MAX_CU_NUM];
	u8                      dpu_cnt;
	u8                      sfm_cnt;
	struct miscdevice       miscdev;
	struct clk              *clk;
#ifdef CONFIG_DEBUG_FS
	struct dentry           *root;
#endif
};

/**
 * struct dpu_buffer_block - DPU buffer block
 * @head: list head
 * @phy_addr: physical address of the blocks memory
 * @vaddr: virtual address of the blocks memory
 * @dma_addr: physical address of the blocks memory
 * @size: total size of the block in bytes
 */
struct dpu_buffer_block {
	struct list_head        head;
	u64                     phy_addr;
	void                    *vaddr;
	dma_addr_t              dma_addr;
	size_t                  capacity;
};

#ifdef CONFIG_DEBUG_FS
static int dpu_debugfs_init(struct xdpu_dev *xdpu);
#endif

/**
 * xdpu_readq - Read 64bit data from the DPU register space
 * @xdpu:	dpu structure
 * @off:	dpu reg offset from base
 *
 * Return:	Returns 64bit value from DPU register specified
 */
static inline __u64 xdpu_readq(struct xdpu_dev *xdpu, u32 off)
{
	u32 low, high;

	low = ioread32(xdpu->regs + off);
	high = ioread32(xdpu->regs + off + 4);

	return low + ((u64)high << 32);
}

/**
 * xdpu_writeq - Write 64bit data to the DPU register space
 * @xdpu:	dpu structure
 * @off:	dpu reg offset from base
 * @val:	64bit data to write
 */
static inline void xdpu_writeq(struct xdpu_dev *xdpu, u32 off, u64 val)
{
	iowrite32(val, xdpu->regs + off);
	iowrite32(val >> 32, xdpu->regs + off + 4);
}

/**
 * xdpu_readl - Read 32bit data from the DPU register space
 * @xdpu:	dpu structure
 * @off:	dpu reg offset from base
 *
 * Return:	Returns 32bit value from DPU register specified
 */
static inline u32 xdpu_readl(struct xdpu_dev *xdpu, u32 off)
{
	return ioread32(xdpu->regs + off);
}

/**
 * xdpu_writel - Write 32bit data to the DPU register space
 * @xdpu:	dpu structure
 * @off:	dpu reg offset from base
 * @val:	64bit data to write
 */
static inline void xdpu_writel(struct xdpu_dev *xdpu, u32 off, u32 val)
{
	iowrite32(val, xdpu->regs + off);
}

/**
 * xlnx_dpu_regs_init - initialize dpu register
 * @xdpu:	dpu structure
 */
static void xlnx_dpu_regs_init(struct xdpu_dev *xdpu)
{
	char cu;

	xdpu_writel(xdpu, DPU_PMU_IP_RST, 0);

	for (cu = 0; cu < xdpu->dpu_cnt; cu++) {
		xdpu_writel(xdpu, DPU_HPBUS(cu), 0x07070f0f);
		xdpu_writel(xdpu, DPU_IPSTART(cu), 0);
	}

	xdpu_writel(xdpu, DPU_INT_ICR, 0xFF);
	udelay(1);
	xdpu_writel(xdpu, DPU_INT_ICR, 0);

	xdpu_writel(xdpu, DPU_PMU_IP_RST, 0xFFFFFFFF);

	xdpu_writel(xdpu, DPU_SFM_INT_DONE, 0);
	xdpu_writel(xdpu, DPU_SFM_CMD_XLEN, 0);
	xdpu_writel(xdpu, DPU_SFM_CMD_YLEN, 0);
	xdpu_writel(xdpu, DPU_SFM_SRC_ADDR, 0);
	xdpu_writel(xdpu, DPU_SFM_DST_ADDR, 0);
	xdpu_writel(xdpu, DPU_SFM_CMD_SCAL, 0);
	xdpu_writel(xdpu, DPU_SFM_CMD_OFF, 0);
	xdpu_writel(xdpu, DPU_SFM_INT_CLR, 0);
	xdpu_writel(xdpu, DPU_SFM_START, 0);
	xdpu_writel(xdpu, DPU_SFM_RESET, 0);
	xdpu_writel(xdpu, DPU_SFM_MODE, 0);
}

/**
 * xlnx_dpu_dump_regs - dump all dpu registers
 * @xdpu:	dpu structure
 */
static void xlnx_dpu_dump_regs(struct xdpu_dev *p)
{
	struct device *dev = p->dev;
	int i;

#define FMT8  "%-27s %08x\n"
#define FMT16 "%-27s %016llx\n"
	dev_warn(dev, "------------[ cut here ]------------\n");
	dev_warn(dev, "Dump DPU Registers:\n");
	dev_info(dev, FMT16, "TARGET_ID", xdpu_readq(p, DPU_TARGETID_L));
	dev_info(dev, FMT8, "PMU_RST", xdpu_readl(p, DPU_PMU_IP_RST));
	dev_info(dev, FMT8, "IP_VER_INFO", xdpu_readl(p, DPU_IPVER_INFO));
	dev_info(dev, FMT8, "IP_FREQENCY", xdpu_readl(p, DPU_IPFREQENCY));
	dev_info(dev, FMT8, "INT_STS", xdpu_readl(p, DPU_INT_STS));
	dev_info(dev, FMT8, "INT_MSK", xdpu_readl(p, DPU_INT_MSK));
	dev_info(dev, FMT8, "INT_RAW", xdpu_readl(p, DPU_INT_RAW));
	dev_info(dev, FMT8, "INT_ICR", xdpu_readl(p, DPU_INT_ICR));
	for (i = 0; i < p->dpu_cnt; i++) {
		dev_warn(dev, "[CU-%d]\n", i);
		dev_info(dev, FMT8, "HPBUS", xdpu_readl(p, DPU_HPBUS(i)));
		dev_info(dev, FMT8, "INSTR", xdpu_readl(p, DPU_INSADDR(i)));
		dev_info(dev, FMT8, "START", xdpu_readl(p, DPU_IPSTART(i)));
		dev_info(dev, FMT16, "ADDR0", xdpu_readq(p, DPU_ADDR0_L(i)));
		dev_info(dev, FMT16, "ADDR1", xdpu_readq(p, DPU_ADDR1_L(i)));
		dev_info(dev, FMT16, "ADDR2", xdpu_readq(p, DPU_ADDR2_L(i)));
		dev_info(dev, FMT16, "ADDR3", xdpu_readq(p, DPU_ADDR3_L(i)));
		dev_info(dev, FMT16, "ADDR4", xdpu_readq(p, DPU_ADDR4_L(i)));
		dev_info(dev, FMT16, "ADDR5", xdpu_readq(p, DPU_ADDR5_L(i)));
		dev_info(dev, FMT16, "ADDR6", xdpu_readq(p, DPU_ADDR6_L(i)));
		dev_info(dev, FMT16, "ADDR7", xdpu_readq(p, DPU_ADDR7_L(i)));
		dev_info(dev, FMT8, "PSTART", xdpu_readl(p, DPU_P_STA_C(i)));
		dev_info(dev, FMT8, "PEND", xdpu_readl(p, DPU_P_END_C(i)));
		dev_info(dev, FMT8, "CSTART", xdpu_readl(p, DPU_C_STA_C(i)));
		dev_info(dev, FMT8, "CEND", xdpu_readl(p, DPU_C_END_C(i)));
		dev_info(dev, FMT8, "SSTART", xdpu_readl(p, DPU_S_STA_C(i)));
		dev_info(dev, FMT8, "SEND", xdpu_readl(p, DPU_S_END_C(i)));
		dev_info(dev, FMT8, "LSTART", xdpu_readl(p, DPU_L_STA_C(i)));
		dev_info(dev, FMT8, "LEND", xdpu_readl(p, DPU_L_END_C(i)));
		dev_info(dev, FMT16, "CYCLE", xdpu_readq(p, DPU_CYCLE_L(i)));
		dev_info(dev, FMT8, "AXI", xdpu_readl(p, DPU_AXI_STS(i)));
	}
	dev_warn(dev, "[SOFTMAX]\n");
	if (p->sfm_cnt) {
#define DUMPREG(r) \
	dev_info(dev, FMT8, #r, ioread32(p->regs + DPU_SFM_##r))
		DUMPREG(INT_DONE);
		DUMPREG(CMD_XLEN);
		DUMPREG(CMD_YLEN);
		DUMPREG(SRC_ADDR);
		DUMPREG(DST_ADDR);
		DUMPREG(CMD_SCAL);
		DUMPREG(CMD_OFF);
		DUMPREG(INT_CLR);
		DUMPREG(START);
		DUMPREG(RESET);
#undef DUMPREG
	}
	dev_warn(dev, "------------[ cut here ]------------\n");
}

/**
 * xlnx_dpu_int_clear - clean DPU interrupt
 * @xdpu:	dpu structure
 * @id:		indicates which cu needs to be clean interrupt
 */
static void xlnx_dpu_int_clear(struct xdpu_dev *xdpu, int id)
{
	xdpu_writel(xdpu, DPU_INT_ICR, BIT(id));
	xdpu_writel(xdpu, DPU_IPSTART(id), 0);
	udelay(1);
	xdpu_writel(xdpu, DPU_INT_ICR, xdpu_readl(xdpu, DPU_INT_ICR)&~BIT(id));
}

/**
 * xlnx_sfm_int_clear - clean softmax interrupt
 * @xdpu:	dpu structure
 */
static void xlnx_sfm_int_clear(struct xdpu_dev *xdpu)
{
	xdpu_writel(xdpu, DPU_SFM_INT_CLR, 1);
	xdpu_writel(xdpu, DPU_SFM_INT_CLR, 0);
}

/**
 * xlnx_dpu_softmax - softmax calculation acceleration using softmax IP
 * @xdpu:	dpu structure
 * @p :	softmax pmeter structure
 *
 * Return:	0 if successful; otherwise -errno
 */
static int xlnx_dpu_softmax(struct xdpu_dev *xdpu, struct ioc_softmax_t *p)
{
	u32 ret = -ETIMEDOUT;
#ifdef DEBUG
	u64 time_start;
#endif

	xdpu_writel(xdpu, DPU_SFM_CMD_XLEN, p->width);
	xdpu_writel(xdpu, DPU_SFM_CMD_YLEN, p->height);

	/* ip limition - softmax supports up to 32-bit addressing */
	xdpu_writel(xdpu, DPU_SFM_SRC_ADDR, p->input);
	xdpu_writel(xdpu, DPU_SFM_DST_ADDR, p->output);

	xdpu_writel(xdpu, DPU_SFM_CMD_SCAL, p->scale);
	xdpu_writel(xdpu, DPU_SFM_CMD_OFF, p->offset);

	xdpu_writel(xdpu, DPU_SFM_RESET, 1);
	xdpu_writel(xdpu, DPU_SFM_MODE, 0);

#ifdef DEBUG
	time_start = ktime_get();
#endif
	/* kickoff softmax */
	xdpu_writel(xdpu, DPU_SFM_START, 1);
	xdpu_writel(xdpu, DPU_SFM_START, 0);

	if (!force_poll && xdpu->cu[xdpu->dpu_cnt].irq != 0) {
		if (!wait_for_completion_timeout(
						 &xdpu->cu[xdpu->dpu_cnt].done,
						 TIMEOUT)) {
			dev_warn(xdpu->dev, "timeout waiting for softmax\n");
			goto err_out;
		}
	} else {
		for (;;) {
			if (xdpu_readl(xdpu, DPU_SFM_INT_DONE) & 0x1) {
				xlnx_sfm_int_clear(xdpu);
				break;
			}
			if (time_after(jiffies, jiffies+TIMEOUT))
				goto err_out;
		}
	}

#ifdef DEBUG
	dev_dbg(xdpu->dev, "%s:  PID=%d CPU=%d TIME=%lldus\n",
		__func__, current->pid, raw_smp_processor_id(),
		ktime_us_delta(ktime_get(), time_start));
#endif

	return 0;
err_out:
	xlnx_dpu_dump_regs(xdpu);
	return ret;
}

/**
 * xlnx_dpu_run - run dpu
 * @xdpu:	dpu structure
 * @p:		dpu run struct, contains the necessary address info
 *
 * Return:	0 if successful; otherwise -errno
 */
static int xlnx_dpu_run(struct xdpu_dev *xdpu, struct ioc_kernel_run_t *p,
			int id)
{
	xdpu_writel(xdpu, DPU_INSADDR(id), p->addr_code >> 12);
	/**
	 * The common regmap:
	 *
	 * Addr0: bias/weights
	 * Addr1: the inter-layer workspacce
	 * Addr2: the 1st input layer
	 * Addr3: the output layer
	 */
	xdpu_writeq(xdpu, DPU_ADDR0_L(id), p->addr0);
	xdpu_writeq(xdpu, DPU_ADDR1_L(id), p->addr1);
	xdpu_writeq(xdpu, DPU_ADDR2_L(id), p->addr2);
	xdpu_writeq(xdpu, DPU_ADDR3_L(id), p->addr3);

	if (p->addr4 != ULLONG_MAX)
		xdpu_writeq(xdpu, DPU_ADDR4_L(id), p->addr4);
	if (p->addr5 != ULLONG_MAX)
		xdpu_writeq(xdpu, DPU_ADDR5_L(id), p->addr5);
	if (p->addr6 != ULLONG_MAX)
		xdpu_writeq(xdpu, DPU_ADDR6_L(id), p->addr6);
	if (p->addr7 != ULLONG_MAX)
		xdpu_writeq(xdpu, DPU_ADDR7_L(id), p->addr7);

	/* kickoff DPU here */
	xdpu_writel(xdpu, DPU_IPSTART(id), 0x1);

#ifdef DEBUG
	p->time_start = ktime_get();
#endif
	if (!force_poll && xdpu->cu[id].irq != 0) {
		if (!wait_for_completion_timeout(&xdpu->cu[id].done,
						 TIMEOUT)) {
			dev_warn(xdpu->dev, "cu[%d] timeout\n", id);
			xlnx_dpu_dump_regs(xdpu);
			return -ETIMEDOUT;
		}
	} else {
		for (;;) {
			if (xdpu_readl(xdpu, DPU_INT_RAW) & BIT(id)) {
				xlnx_dpu_int_clear(xdpu, id);
				break;
			}
			if (time_after(jiffies, jiffies+TIMEOUT)) {
				dev_warn(xdpu->dev, "cu[%d] timeout", id);
				xlnx_dpu_dump_regs(xdpu);
				return -ETIMEDOUT;
			}
		}
	}

	p->core_id = id;
	p->pend_cnt = xdpu_readl(xdpu, DPU_P_END_C(id));
	p->cend_cnt = xdpu_readl(xdpu, DPU_C_END_C(id));
	p->send_cnt = xdpu_readl(xdpu, DPU_S_END_C(id));
	p->lend_cnt = xdpu_readl(xdpu, DPU_L_END_C(id));
	p->pstart_cnt = xdpu_readl(xdpu, DPU_P_STA_C(id));
	p->cstart_cnt = xdpu_readl(xdpu, DPU_C_STA_C(id));
	p->sstart_cnt = xdpu_readl(xdpu, DPU_S_STA_C(id));
	p->lstart_cnt = xdpu_readl(xdpu, DPU_L_STA_C(id));
	p->counter = xdpu_readq(xdpu, DPU_CYCLE_L(id));

#ifdef DEBUG
	p->time_end = ktime_get();
	dev_dbg(xdpu->dev,
		"%s:  PID=%d DPU=%d CPU=%d TIME=%lldms complete!\n",
		__func__, current->pid, id, raw_smp_processor_id(),
		ktime_ms_delta(p->time_end, p->time_start));
#endif
	return 0;
}

/**
 * xlnx_dpu_alloc_bo - alloc contiguous physical memory for dpu
 * @xdpu:	dpu structure
 * @req:	dpcma_req_alloc struct, contains the request info
 *
 * Return:	0 if successful; otherwise -errno
 */
static long xlnx_dpu_alloc_bo(struct xdpu_dev *xdpu,
			      struct dpcma_req_alloc *req)
{
	int ret = 0;
	size_t size = 0;
	struct dpu_buffer_block *pb = NULL;

	pb = kzalloc(sizeof(struct dpu_buffer_block), GFP_KERNEL);
	if (!pb)
		return -ENOMEM;

	size = req->size;
	/*
	if (get_user(size, &req->size)) {
		kfree(pb);
		return -EFAULT;
	}
	*/

	pb->capacity = ALIGN(size, PAGE_SIZE);

	req->capacity = pb->capacity;
	/*	
	if (put_user(pb->capacity, &req->capacity)) {
		kfree(pb);
		return -EFAULT;
	}
	*/

	pb->vaddr = dma_alloc_coherent(xdpu->dev, pb->capacity, &pb->dma_addr,
				       GFP_KERNEL);
	if (!pb->vaddr) {
		kfree(pb);
		return -ENOMEM;
	}

	pb->phy_addr = pb->dma_addr;
	req->phy_addr = pb->phy_addr;
	/*
	if (put_user(pb->phy_addr, &req->phy_addr))
		goto err_out;
	*/

	list_add(&pb->head, &xdpu->head);

	return ret;

err_out:
	dma_free_coherent(xdpu->dev, pb->capacity, pb->vaddr,
		     pb->dma_addr);
	kfree(pb);
	return -EFAULT;
}

/**
 * xlnx_dpu_free_bo - free contiguous physical memory allocated
 * @xdpu:	dpu structure
 * @req:	dpcma_req_free struct, contains the request info
 *
 * Return:	0 if successful; otherwise -errno
 */
static long xlnx_dpu_free_bo(struct xdpu_dev *xdpu, struct dpcma_req_free *req)
{
	struct list_head *pos = NULL;
	struct list_head *next = NULL;
	u64 phy_addr = 0;
	int ret = 0;
	struct dpu_buffer_block *h;

	phy_addr = req->phy_addr;
	/*
	if (get_user(phy_addr, &req->phy_addr))
		return -EFAULT;
	*/

	list_for_each_safe(pos, next, &xdpu->head) {
		h = list_entry(pos, struct dpu_buffer_block, head);
		if (phy_addr == h->phy_addr) {
			dma_free_coherent(xdpu->dev, h->capacity, h->vaddr,
					  h->dma_addr);
			list_del(pos);
			kfree(h);
		}
	}

	return ret;
}

/**
 * xlnx_dpu_sync_bo - flush/invalidate cache for allocated memory
 * @xdpu:	dpu structure
 * @req:	dpcma_req_sync struct, contains the request info
 *
 * Return:	0 if successful; otherwise -errno
 */
static long xlnx_dpu_sync_bo(struct xdpu_dev *xdpu, struct dpcma_req_sync *req)
{
	struct list_head *pos = NULL;
	long phy_addr = 0;
	int dir = 0;
	size_t size = 0;
	int ret = 0;
	size_t offset = 0;
	struct dpu_buffer_block *h;

	phy_addr = req->phy_addr;
	size = req->size;
	dir = req->direction;
	/*
	if (get_user(phy_addr, &req->phy_addr) ||
	    get_user(size, &req->size) ||
	    get_user(dir, &req->direction))
		return -EFAULT;
	*/

	if (dir != DPCMA_FROM_CPU_TO_DEVICE &&
	    dir != DPCMA_FROM_DEVICE_TO_CPU) {
		dev_err(xdpu->dev, "invalid direction. direction = %d\n", dir);
		return -EINVAL;
	}

	list_for_each(pos, &xdpu->head) {
		h = list_entry(pos, struct dpu_buffer_block, head);
		if (phy_addr >= h->phy_addr &&
		    phy_addr < h->phy_addr + h->capacity) {
			offset = h->dma_addr + phy_addr - h->phy_addr;
			if (dir == DPCMA_FROM_DEVICE_TO_CPU)
				dma_sync_single_for_cpu(xdpu->dev,
							offset,
							size,
							DMA_FROM_DEVICE);
			else if (dir == DPCMA_FROM_CPU_TO_DEVICE)
				dma_sync_single_for_device(xdpu->dev,
							   offset,
							   size,
							   DMA_TO_DEVICE);
		}
	}
	return ret;
}

/**
 * xlnx_dpu_ioctl - control ioctls for the DPU
 * @file:	file handle of the DPU device
 * @cmd:	ioctl code
 * @arg:	pointer to user passsed structure
 *
 * Return:	0 if successful; otherwise -errno
 */
static long xlnx_dpu_ioctl(struct file *file, unsigned int cmd,
			   unsigned long arg)
{
	int ret = 0;
	struct xdpu_dev *xdpu;
	void __user *data = NULL;
	int rval = -EINVAL;

	xdpu = container_of(file->private_data, struct xdpu_dev, miscdev);

	if (_IOC_TYPE(cmd) != DPU_IOC_MAGIC)
		return -ENOTTY;

	/* check if ioctl argument is present and valid */
	if (_IOC_DIR(cmd) != _IOC_NONE) {
		data = (void __user *)arg;
		if (!data)
			return rval;
	}

	switch (cmd) {
	case DPUIOC_RUN:
	{
		struct ioc_kernel_run_t t;
		int id;
		memcpy(&t, (void *)arg, sizeof(struct ioc_kernel_run_t));
		/*
		if (copy_from_user(&t, (void *)arg,
				   sizeof(struct ioc_kernel_run_t))) {
			return -EINVAL;
		}
		*/
		id = t.core_id;
		if (id >= xdpu->dpu_cnt)
			return -EINVAL;

		dev_dbg(xdpu->dev,
			"%s PID=%d DPU=%d CPU=%d Comm=%.20s waiting",
			__func__, current->pid, id, raw_smp_processor_id(),
			current->comm);

		/* Allows one process to run the cu by using a mutex */
		mutex_lock(&xdpu->cu[id].mutex);

		ret = xlnx_dpu_run(xdpu, &t, id);

		mutex_unlock(&xdpu->cu[id].mutex);
		
		memcpy(data, &t, sizeof(struct ioc_kernel_run_t));
		/*
		if (copy_to_user(data, &t, sizeof(struct ioc_kernel_run_t)))
			return -EINVAL;
		*/
		break;
	}
	case DPUIOC_CREATE_BO:
		return xlnx_dpu_alloc_bo(xdpu, (struct dpcma_req_alloc *)arg);
	case DPUIOC_FREE_BO:
		return xlnx_dpu_free_bo(xdpu, (struct dpcma_req_free *)arg);
	case DPUIOC_SYNC_BO:
		return xlnx_dpu_sync_bo(xdpu, (struct dpcma_req_sync *)arg);
	case DPUIOC_G_INFO:
	{
		u32 dpu_info = xdpu_readl(xdpu, DPU_IPVER_INFO);

		memcpy(data, &dpu_info, sizeof(dpu_info));
		/*
		if (copy_to_user(data, &dpu_info, sizeof(dpu_info)))
			return -EFAULT;
		*/
		break;
	}
	case DPUIOC_G_TGTID:
	{
		u64 fingerprint = xdpu_readq(xdpu, DPU_TARGETID_L);

		memcpy(data, &fingerprint, sizeof(fingerprint));
		/*
		if (copy_to_user(data, &fingerprint, sizeof(fingerprint)))
			return -EFAULT;
		*/
		break;
	}
	case DPUIOC_RUN_SOFTMAX:
	{
		struct ioc_softmax_t t;

		memcpy(&t, (void *)arg, sizeof(struct ioc_softmax_t));
		/*
		if (copy_from_user(&t, (void *)arg,
				   sizeof(struct ioc_softmax_t))) {
			dev_err(xdpu->dev, "copy_from_user softmax_t fail\n");
			return -EINVAL;
		}
		*/
		/* Allows one process to run the softmax by using a mutex */
		mutex_lock(&xdpu->cu[xdpu->dpu_cnt].mutex);

		ret = xlnx_dpu_softmax(xdpu, &t);

		mutex_unlock(&xdpu->cu[xdpu->dpu_cnt].mutex);

		break;
	}
	case DPUIOC_REG_READ:
	{
		u32 val = 0;
		u32 off = 0;

		memcpy(&off, (void *)arg, sizeof(off));
		/*
		if (copy_from_user(&off, (void *)arg, sizeof(off))) {
			dev_err(xdpu->dev, "copy_from_user off failed\n");
			return -EINVAL;
		}
		*/
		val = xdpu_readl(xdpu, off);

		memcpy(data, &val, sizeof(val));
		/*
		if (copy_to_user(data, &val, sizeof(val)))
			return -EFAULT;
		*/
		break;
	}
	default:
		ret = -EOPNOTSUPP;
	}

	return ret;
}

/**
 * xlnx_dpu_isr - interrupt handler for DPU.
 * @irq:	Interrupt number.
 * @data:	DPU device structure.
 *
 * Return: IRQ_HANDLED.
 *
 */
static irqreturn_t xlnx_dpu_isr(int irq, void *data)
{
	struct xdpu_dev *xdpu = data;
	int i = 0;

	for (i = 0; i < xdpu->dpu_cnt; i++) {
		if (irq == xdpu->cu[i].irq) {
			xlnx_dpu_int_clear(xdpu, i);
			dev_dbg(xdpu->dev, "%s:  DPU=%d IRQ=%d",
				__func__, i, irq);
			complete(&xdpu->cu[i].done);
		}
	}

	if (irq == xdpu->cu[xdpu->dpu_cnt].irq) {
		xlnx_sfm_int_clear(xdpu);
		dev_dbg(xdpu->dev, "%s:  softmax IRQ=%d", __func__, irq);
		complete(&xdpu->cu[xdpu->dpu_cnt].done);
	}


	return IRQ_HANDLED;
}

static int xlnx_dpu_open(struct inode *inode, struct file *filp)
{
	return 0;
}

static int xlnx_dpu_release(struct inode *inode, struct file *filp)
{
	return 0;
}

/*
 * xlnx_dpu_mmap - maps cma ranges into userspace
 * @file:	file structure for the device
 * @vma:	VMA to map the registers into
 *
 * Return:	0 if successful; otherwise -errno
 */
static int xlnx_dpu_mmap(struct file *file, struct vm_area_struct *vma)
{
	size_t size = vma->vm_end - vma->vm_start;
	phys_addr_t offset = (phys_addr_t)vma->vm_pgoff << PAGE_SHIFT;

	if (offset >> PAGE_SHIFT != vma->vm_pgoff)
		return -EINVAL;

	if (offset + (phys_addr_t)size - 1 < offset)
		return -EINVAL;

	if (!(vma->vm_pgoff + size <= __pa(high_memory)))
		return -EINVAL;

	if (remap_pfn_range(vma,
			    vma->vm_start,
			    vma->vm_pgoff,
			    size,
			    vma->vm_page_prot)) {
		return -EAGAIN;
	}

	return 0;
}

const static struct file_operations dev_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = xlnx_dpu_ioctl,
	.mmap = xlnx_dpu_mmap,
	.open = xlnx_dpu_open,
	.release = xlnx_dpu_release,
};

/**
 * get_irq - get irq
 * @pdev:	dpu platform device
 * @xdpu:	dpu structure
 *
 * Return:	0 if successful; otherwise -errno
 */
static u32 get_irq(struct platform_device *pdev, struct xdpu_dev *xdpu)
{
	u8 i;
	u32 ret = 0;
	u8 sfm_no = xdpu->dpu_cnt;
	struct device *dev = xdpu->dev;

	if (force_poll) {
		dev_warn(dev, "no IRQ, using polling mode\n");
		return ret;
	}

	if ((xdpu->dpu_cnt + xdpu->sfm_cnt) != platform_irq_count(pdev))
		dev_warn(dev, "IRQs num(%d) doesn't match dpu/sfm num(%d)!\n",
			platform_irq_count(pdev),
			xdpu->dpu_cnt + xdpu->sfm_cnt);

	for (i = 0; i < xdpu->dpu_cnt; i++) {
		xdpu->cu[i].irq = platform_get_irq(pdev, i);
		if (xdpu->cu[i].irq < 0)
			return -EINVAL;

		/* DPU interrupt: level, active high */
		irq_set_irq_type(xdpu->cu[i].irq,
				 IRQ_TYPE_LEVEL_HIGH);

		ret = devm_request_irq(dev,
				       xdpu->cu[i].irq,
				       xlnx_dpu_isr,
				       0,
				       devm_kasprintf(dev,
						      GFP_KERNEL,
						      "%s-cu[%d]",
						      dev_name(dev),
						      i),
				       xdpu);
		if (ret < 0)
			return ret;
	}

	if (xdpu->sfm_cnt) {
		xdpu->cu[sfm_no].irq = platform_get_irq(pdev, sfm_no);
		if (xdpu->cu[sfm_no].irq < 0)
			return -EINVAL;

		/* DPU interrupt: level, active high */
		irq_set_irq_type(xdpu->cu[sfm_no].irq,
				 IRQ_TYPE_LEVEL_HIGH);
		ret = devm_request_irq(dev,
				       xdpu->cu[sfm_no].irq,
				       xlnx_dpu_isr,
				       0,
				       devm_kasprintf(dev,
						      GFP_KERNEL,
						      "%s-softmax",
						      dev_name(dev)),
				       xdpu);
		if (ret < 0)
			return ret;
	}

	return ret;
}

/**
 * xlnx_dpu_parse_of - Parse device tree information
 * @xdpu: Pointer to dpu device structure
 *
 * This function reads the device tree contents
 *
 * Return: 0 on success. -EINVAL for invalid value.
 */
static int xlnx_dpu_parse_of(struct xdpu_dev *xdpu)
{
	int ret;
	struct clk *dpu_clk;
	struct clk *dsp_clk;

	xdpu->clk = devm_clk_get(xdpu->dev, "s_axi_aclk");
	if (IS_ERR(xdpu->clk)) {
		if (PTR_ERR(xdpu->clk) != -ENOENT)
			return PTR_ERR(xdpu->clk);

		/*
		 * Clock framework support is optional, continue on,
		 * anyways if we don't find a matching clock
		 */
		xdpu->clk = NULL;
	}

	ret = clk_prepare_enable(xdpu->clk);
	if (ret) {
		dev_err(xdpu->dev, "failed to enable s_axi_aclk(%d)\n", ret);
		return ret;
	}

	dpu_clk = devm_clk_get(xdpu->dev, "m_axi_dpu_aclk");
	if (IS_ERR(dpu_clk))
		dpu_clk = NULL;

	dsp_clk = devm_clk_get(xdpu->dev, "dpu_2x_clk");
	if (IS_ERR(dsp_clk))
		dsp_clk = NULL;

	/* currently, DTG doesn't create clock for dpu in microblaze */
	if (xdpu->clk && dpu_clk && dsp_clk)
		dev_dbg(xdpu->dev,
			"Freq: axilite: %lu MHz, dpu: %lu MHz, dsp: %lu MHz",
			clk_get_rate(xdpu->clk)/1000000,
			clk_get_rate(dpu_clk)/1000000,
			clk_get_rate(dsp_clk)/1000000);

	return 0;
}

/**
 * xlnx_dpu_probe - probe dpu device
 * @pdev: Pointer to dpu platform device structure
 *
 * Return: 0 on success. -EINVAL for invalid value.
 */
static int xlnx_dpu_probe(struct platform_device *pdev)
{
	int i = 0;
	struct xdpu_dev *xdpu;
	struct resource *res;
	struct device *dev;
	u32 val;

	xdpu = devm_kzalloc(&pdev->dev, sizeof(*xdpu), GFP_KERNEL);
	if (!xdpu)
		return -ENOMEM;

	xdpu->dev = &pdev->dev;
	dev = xdpu->dev;

	xdpu->regs = devm_platform_get_and_ioremap_resource(pdev, 0, &res);
	if (IS_ERR(xdpu->regs))
		return -ENOMEM;

	val = xdpu_readl(xdpu, DPU_IPVER_INFO);
	xdpu->dpu_cnt = DPU_NUM(val);
	xdpu->sfm_cnt = SFM_NUM(val);

	if (!(DPU_VER(val) && DPU_VER(val) >= 0x34)) {
		dev_err(dev, "DPU IP need upgrade to 3.4 or later");
		return -EINVAL;
	}

	if (xlnx_dpu_parse_of(xdpu))
		return -EINVAL;

	if (get_irq(pdev, xdpu))
		return -EINVAL;

	of_reserved_mem_device_init(dev);

	/* Vivadoflow DPU ip is capable of 40-bit physical addresses only */
	if (dma_set_mask_and_coherent(dev, DMA_BIT_MASK(40))) {
		/* fall back to 32-bit DMA mask */
		if (dma_set_mask_and_coherent(dev, DMA_BIT_MASK(32)))
			return -EINVAL;
	}

	for (i = 0; i < xdpu->dpu_cnt + xdpu->sfm_cnt; i++) {
		init_completion(&xdpu->cu[i].done);
		mutex_init(&xdpu->cu[i].mutex);
	}

	INIT_LIST_HEAD(&xdpu->head);

	/* Save driver private data */
	platform_set_drvdata(pdev, xdpu);

	xdpu->miscdev.minor  = MISC_DYNAMIC_MINOR;
	xdpu->miscdev.name   = DEVICE_NAME;
	xdpu->miscdev.fops   = &dev_fops;
	xdpu->miscdev.parent = dev;

	if (misc_register(&xdpu->miscdev)) {
		kfree(xdpu);
		return -EINVAL;
	}

	xlnx_dpu_regs_init(xdpu);

#ifdef CONFIG_DEBUG_FS
	dpu_debugfs_init(xdpu);
#endif

	val = xdpu_readl(xdpu, DPU_IPFREQENCY);

	dev_info(dev,
		 "found %d dpu @%dMHz and %d softmax, dpu registered as /dev/dpu successfully",
		 xdpu->dpu_cnt, DPU_FREQ(val), xdpu->sfm_cnt);

	return 0;
}

/**
 * xlnx_dpu_remove - clean up structures
 * @pdev:	The structure containing the device's details
 *
 * Return: 0
 */
static int xlnx_dpu_remove(struct platform_device *pdev)
{
	struct xdpu_dev *xdpu = platform_get_drvdata(pdev);
	int i = 0;

	if (!xdpu)
		return -EINVAL;

	/* clean all regs */
	for (i = 0; i < 0x200; i++)
		iowrite32(0, xdpu->regs + i*4);

#ifdef CONFIG_DEBUG_FS
	debugfs_remove_recursive(xdpu->root);
#endif
	platform_set_drvdata(pdev, NULL);
	misc_deregister(&xdpu->miscdev);

	dev_info(xdpu->dev, "%s: device /dev/dpu unregistered\n", __func__);
	return 0;
}

#ifdef CONFIG_OF
static const struct of_device_id dpu_of_match[] = {
	{ .compatible = "xlnx,dpuczdx8g-3.4" },
	{ /* end of table */ }
};
MODULE_DEVICE_TABLE(of, dpu_of_match);
#endif

static struct platform_driver xlnx_dpu_drv = {
	.probe = xlnx_dpu_probe,
	.remove = xlnx_dpu_remove,

	.driver = {
		.name = DRV_NAME,
#ifdef CONFIG_OF
		.of_match_table = dpu_of_match,
#endif
	},
};

/**
 * xlnx_dpu_init - Registers the driver
 *
 * Return: 0 on success, -1 on allocation error
 *
 * Registers the dpu driver
 */
static int __init xlnx_dpu_init(void)
{
	pr_info(DRV_NAME ": " DRIVER_DESC "\n");

	return platform_driver_register(&xlnx_dpu_drv);
}

/**
 * xlnx_dpu_exit - Destroys the driver
 *
 * Unregisters the dpu driver
 */
static void __exit xlnx_dpu_exit(void)
{
	platform_driver_unregister(&xlnx_dpu_drv);
}

module_init(xlnx_dpu_init);
module_exit(xlnx_dpu_exit);

MODULE_DESCRIPTION(DRIVER_DESC);
MODULE_AUTHOR("Ye Yang <yey@xilinx.com>");
MODULE_LICENSE("GPL v2");
MODULE_ALIAS("platform:" DRV_NAME);

#ifdef CONFIG_DEBUG_FS

#define dump_register(n)			\
{						\
	.name	= #n,				\
	.offset	= DPU_##n,				\
}

static const struct debugfs_reg32 cu_regs[4][38] = {
	{
	dump_register(IPVER_INFO),
	dump_register(IPFREQENCY),
	dump_register(TARGETID_L),
	dump_register(TARGETID_H),
	dump_register(IPSTART(0)),
	dump_register(INSADDR(0)),
	dump_register(ADDR0_L(0)),
	dump_register(ADDR0_H(0)),
	dump_register(ADDR1_L(0)),
	dump_register(ADDR1_H(0)),
	dump_register(ADDR2_L(0)),
	dump_register(ADDR2_H(0)),
	dump_register(ADDR3_L(0)),
	dump_register(ADDR3_H(0)),
	dump_register(ADDR4_L(0)),
	dump_register(ADDR4_H(0)),
	dump_register(ADDR5_L(0)),
	dump_register(ADDR5_H(0)),
	dump_register(ADDR6_L(0)),
	dump_register(ADDR6_H(0)),
	dump_register(ADDR7_L(0)),
	dump_register(ADDR7_H(0)),
	dump_register(CYCLE_L(0)),
	dump_register(CYCLE_H(0)),
	dump_register(P_STA_C(0)),
	dump_register(P_END_C(0)),
	dump_register(C_STA_C(0)),
	dump_register(C_END_C(0)),
	dump_register(S_STA_C(0)),
	dump_register(S_END_C(0)),
	dump_register(L_STA_C(0)),
	dump_register(L_END_C(0)),
	dump_register(AXI_STS(0)),
	dump_register(HPBUS(0)),
	dump_register(INT_STS),
	dump_register(INT_MSK),
	dump_register(INT_RAW),
	dump_register(INT_ICR),
	},
	{
	dump_register(IPVER_INFO),
	dump_register(IPFREQENCY),
	dump_register(TARGETID_L),
	dump_register(TARGETID_H),
	dump_register(IPSTART(1)),
	dump_register(INSADDR(1)),
	dump_register(ADDR0_L(1)),
	dump_register(ADDR0_H(1)),
	dump_register(ADDR1_L(1)),
	dump_register(ADDR1_H(1)),
	dump_register(ADDR2_L(1)),
	dump_register(ADDR2_H(1)),
	dump_register(ADDR3_L(1)),
	dump_register(ADDR3_H(1)),
	dump_register(ADDR4_L(1)),
	dump_register(ADDR4_H(1)),
	dump_register(ADDR5_L(1)),
	dump_register(ADDR5_H(1)),
	dump_register(ADDR6_L(1)),
	dump_register(ADDR6_H(1)),
	dump_register(ADDR7_L(1)),
	dump_register(ADDR7_H(1)),
	dump_register(CYCLE_L(1)),
	dump_register(CYCLE_H(1)),
	dump_register(P_STA_C(1)),
	dump_register(P_END_C(1)),
	dump_register(C_STA_C(1)),
	dump_register(C_END_C(1)),
	dump_register(S_STA_C(1)),
	dump_register(S_END_C(1)),
	dump_register(L_STA_C(1)),
	dump_register(L_END_C(1)),
	dump_register(AXI_STS(1)),
	dump_register(HPBUS(1)),
	dump_register(INT_STS),
	dump_register(INT_MSK),
	dump_register(INT_RAW),
	dump_register(INT_ICR),
	},
	{
	dump_register(IPVER_INFO),
	dump_register(IPFREQENCY),
	dump_register(TARGETID_L),
	dump_register(TARGETID_H),
	dump_register(IPSTART(2)),
	dump_register(INSADDR(2)),
	dump_register(ADDR0_L(2)),
	dump_register(ADDR0_H(2)),
	dump_register(ADDR1_L(2)),
	dump_register(ADDR1_H(2)),
	dump_register(ADDR2_L(2)),
	dump_register(ADDR2_H(2)),
	dump_register(ADDR3_L(2)),
	dump_register(ADDR3_H(2)),
	dump_register(ADDR4_L(2)),
	dump_register(ADDR4_H(2)),
	dump_register(ADDR5_L(2)),
	dump_register(ADDR5_H(2)),
	dump_register(ADDR6_L(2)),
	dump_register(ADDR6_H(2)),
	dump_register(ADDR7_L(2)),
	dump_register(ADDR7_H(2)),
	dump_register(CYCLE_L(2)),
	dump_register(CYCLE_H(2)),
	dump_register(P_STA_C(2)),
	dump_register(P_END_C(2)),
	dump_register(C_STA_C(2)),
	dump_register(C_END_C(2)),
	dump_register(S_STA_C(2)),
	dump_register(S_END_C(2)),
	dump_register(L_STA_C(2)),
	dump_register(L_END_C(2)),
	dump_register(AXI_STS(2)),
	dump_register(HPBUS(2)),
	dump_register(INT_STS),
	dump_register(INT_MSK),
	dump_register(INT_RAW),
	dump_register(INT_ICR),
	},
	{
	dump_register(IPVER_INFO),
	dump_register(IPFREQENCY),
	dump_register(TARGETID_L),
	dump_register(TARGETID_H),
	dump_register(IPSTART(3)),
	dump_register(INSADDR(3)),
	dump_register(ADDR0_L(3)),
	dump_register(ADDR0_H(3)),
	dump_register(ADDR1_L(3)),
	dump_register(ADDR1_H(3)),
	dump_register(ADDR2_L(3)),
	dump_register(ADDR2_H(3)),
	dump_register(ADDR3_L(3)),
	dump_register(ADDR3_H(3)),
	dump_register(ADDR4_L(3)),
	dump_register(ADDR4_H(3)),
	dump_register(ADDR5_L(3)),
	dump_register(ADDR5_H(3)),
	dump_register(ADDR6_L(3)),
	dump_register(ADDR6_H(3)),
	dump_register(ADDR7_L(3)),
	dump_register(ADDR7_H(3)),
	dump_register(CYCLE_L(3)),
	dump_register(CYCLE_H(3)),
	dump_register(P_STA_C(3)),
	dump_register(P_END_C(3)),
	dump_register(C_STA_C(3)),
	dump_register(C_END_C(3)),
	dump_register(S_STA_C(3)),
	dump_register(S_END_C(3)),
	dump_register(L_STA_C(3)),
	dump_register(L_END_C(3)),
	dump_register(AXI_STS(3)),
	dump_register(HPBUS(3)),
	dump_register(INT_STS),
	dump_register(INT_MSK),
	dump_register(INT_RAW),
	dump_register(INT_ICR),
	},
};

static const struct debugfs_reg32 sfm_regs[] = {
	dump_register(IPVER_INFO),
	dump_register(IPFREQENCY),
	dump_register(TARGETID_L),
	dump_register(TARGETID_H),
	dump_register(SFM_INT_DONE),
	dump_register(SFM_CMD_XLEN),
	dump_register(SFM_CMD_YLEN),
	dump_register(SFM_SRC_ADDR),
	dump_register(SFM_DST_ADDR),
	dump_register(SFM_CMD_SCAL),
	dump_register(SFM_CMD_OFF),
	dump_register(SFM_INT_CLR),
	dump_register(SFM_START),
	dump_register(SFM_RESET),
	dump_register(SFM_MODE),
	dump_register(INT_STS),
	dump_register(INT_MSK),
	dump_register(INT_RAW),
	dump_register(INT_ICR),
};

static int dump_show(struct seq_file *seq, void *v)
{
	struct xdpu_dev *xdpu = seq->private;
	struct dpu_buffer_block *h;
	static const char units[] = "KMG";
	const char *unit = units;
	unsigned long delta = 0;

	seq_puts(seq,
		 "Virtual Address\t\t\t\tRequest Mem\t\tPhysical Address\n");
	list_for_each_entry(h, &xdpu->head, head) {
		delta = (h->capacity) >> 10;
		while (!(delta & 1023) && unit[1]) {
			delta >>= 10;
			unit++;
		}
		seq_printf(seq, "%px-%px   %9lu%c         %016llx-%016llx\n",
			   h->vaddr, h->vaddr+h->capacity, delta, *unit,
			   (u64)h->dma_addr, (u64)(h->dma_addr+h->capacity));
		delta = 0;
		unit = units;
	}

	return 0;
}
DEFINE_SHOW_ATTRIBUTE(dump);

/**
 * dpu_debugfs_init - create DPU debugfs directory.
 * @xdpu:	dpu structure
 *
 * Create DPU debugfs directory. Returns zero in case of success and a negative
 * error code in case of failure.
 */
static int dpu_debugfs_init(struct xdpu_dev *xdpu)
{
	char buf[32];
	struct debugfs_regset32 *regset;
	struct dentry *dentry;
	int i = 0;

	xdpu->root = debugfs_create_dir("dpu", NULL);
	if (IS_ERR(xdpu->root)) {
		dev_err(xdpu->dev, "failed to create debugfs root\n");
		return -ENODEV;
	}

	debugfs_create_file("dma_pool", 0444, xdpu->root, xdpu, &dump_fops);

	for (i = 0; i < xdpu->dpu_cnt; i++) {
		if (snprintf(buf, 32, "cu-%d", i) < 0)
			return -EINVAL;
		dentry = debugfs_create_dir(buf, xdpu->root);
		regset = devm_kzalloc(xdpu->dev, sizeof(*regset), GFP_KERNEL);
		if (!regset)
			return -ENOMEM;
		regset->regs = cu_regs[i];
		regset->nregs = ARRAY_SIZE(cu_regs[i]);
		regset->base = xdpu->regs;
		debugfs_create_regset32("registers", 0444, dentry, regset);

	}

	if (xdpu->sfm_cnt) {
		dentry = debugfs_create_dir("softmax", xdpu->root);
		regset = devm_kzalloc(xdpu->dev, sizeof(*regset), GFP_KERNEL);
		if (!regset)
			return -ENOMEM;
		regset->regs = sfm_regs;
		regset->nregs = ARRAY_SIZE(sfm_regs);
		regset->base = xdpu->regs;
		debugfs_create_regset32("registers", 0444, dentry, regset);
	}
	return 0;
}
#endif
