/**
	@brief Source file of vendor ai net sample code.

	@file ai3_net.c

	@ingroup ai_net_sample

	@note Nothing.

	Copyright Novatek Microelectronics Corp. 2020.  All rights reserved.
*/

/*-----------------------------------------------------------------------------*/
/* Including Files                                                             */
/*-----------------------------------------------------------------------------*/
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "hd_gfx.h"
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "hdal.h"
#include "hd_debug.h"
#include "vendor_ai.h"
#include "vendor_ai_cpu/vendor_ai_cpu.h"
#include "ai_cpu_post_accuracy.h"

#if defined(_NVT_NVR_SDK_)
#include <comm/nvtmem_if.h>
#include <sys/ioctl.h>
#endif

#if defined(_BSP_NA51068_) || defined(_BSP_NA51090_)
#include "vendor_common.h"
#endif

// platform dependent
#if defined(__LINUX)
#include <signal.h>
#include <pthread.h>			//for pthread API
#define MAIN(argc, argv) 		int main(int argc, char** argv)
#define GETCHAR()				getchar()
#else
#include <FreeRTOS_POSIX.h>
#include <FreeRTOS_POSIX/signal.h>
#include <FreeRTOS_POSIX/pthread.h> //for pthread API
#include <kwrap/util.h>		//for sleep API
#define sleep(x)    			vos_util_delay_ms(1000*(x))
#define msleep(x)    			vos_util_delay_ms(x)
#define usleep(x)   			vos_util_delay_us(x)
#include <kwrap/examsys.h> 	//for MAIN(), GETCHAR() API
#define MAIN(argc, argv) 		EXAMFUNC_ENTRY(ai3_net, argc, argv)
#define GETCHAR()				NVT_EXAMSYS_GETCHAR()
#endif

#define DEBUG_MENU 					1

//dump net-info
#define DBG_IO_BUF_INFO_DUMP        0  // dump in/out buffer info before open()

//dump net-version
#define DBG_MODEL_VERSION			0  // dump model gen version before open()

//dump net i/o buffer inside proc
#define DBG_OUT_DUMP                0  // debug mode, dump output buffer of each layer

//dump performance inside proc
#define DBG_TIME_DUMP               0  // debug mode, dump perf time (us) for each layer
#define DBG_BW_DUMP                 0  // debug mode, dump perf bandwidth (bytes) for each layer
#define DBG_PERF_DUMP               0  // debug mode, dump perf time (us) and bandwidth (bytes) and its ratio for each layer

//dump performance with begin-end range
#define DBG_PERF_TIME_UT            0  // perf time(us) and ut(%) for each proc()
#define DBG_PERF_TIME_UT2           0  // perf time(us) and ut(%) for each second

//dump net result
#define DUMP_POSTPROC_INFO			1  // dump result buffer info after proc()

//AI3 List
#define YUV_CROP			        0  // Set ai3_buf list


///////////////////////////////////////////////////////////////////////////////

#define USE_DDR						DDR_ID0 //DDR_ID1

#define FREE_RUN					0

#define NET_PATH_ID					UINT32

#define VENDOR_AI_CFG  				0x000f0000  //vendor ai config

#define AI_RGB_BUFSIZE(w, h)		(ALIGN_CEIL_4((w) * HD_VIDEO_PXLFMT_BPP(HD_VIDEO_PXLFMT_RGB888_PLANAR) / 8) * (h))

#define NET_VDO_SIZE_W	1920 //max for net
#define NET_VDO_SIZE_H	1080 //max for net

///////////////////////////////////////////////////////////////////////////////

/*-----------------------------------------------------------------------------*/
/* Type Definitions                                                            */
/*-----------------------------------------------------------------------------*/

typedef struct _MEM_PARM {
	UINTPTR pa;
	UINTPTR va;
	UINT32 size;
	UINTPTR blk;
} MEM_PARM;

/*-----------------------------------------------------------------------------*/
/* Global Functions                                                             */
/*-----------------------------------------------------------------------------*/


static HD_RESULT mem_alloc(MEM_PARM *mem_parm, CHAR *name, UINT32 size)
{
	HD_RESULT ret = HD_OK;
	UINTPTR pa   = 0;
	void  *va   = NULL;

	//alloc private pool
	ret = hd_common_mem_alloc(name, &pa, (void **)&va, size, USE_DDR);
	if (ret != HD_OK) {
		return ret;
	}

	mem_parm->pa   = pa;
	mem_parm->va   = (UINTPTR)va;
	mem_parm->size = size;
	mem_parm->blk  = (UINTPTR)-1;

	return HD_OK;
}

static HD_RESULT mem_free(MEM_PARM *mem_parm)
{
	HD_RESULT ret = HD_OK;

	//free private pool
	ret =  hd_common_mem_free(mem_parm->pa, (void *)mem_parm->va);
	if (ret != HD_OK) {
		return ret;
	}

	mem_parm->pa = 0;
	mem_parm->va = 0;
	mem_parm->size = 0;
	mem_parm->blk = (UINT32)-1;

	return HD_OK;
}

/*-----------------------------------------------------------------------------*/
/* Input Functions                                                             */
/*-----------------------------------------------------------------------------*/

///////////////////////////////////////////////////////////////////////////////

typedef struct _NET_IN_CONFIG {

	CHAR input_filename[256];
	UINT32 w;
	UINT32 h;
	UINT32 c;
	UINT32 loff;
	UINT32 fmt;

} NET_IN_CONFIG;

typedef struct _NET_IN {

	NET_IN_CONFIG in_cfg;
	MEM_PARM input_mem;
	UINT32 in_id;
	VENDOR_AI3_BUF src_img;

} NET_IN;

static NET_IN g_in[16] = {0};

HD_RESULT scale_image(HD_GFX_IMG_BUF *src_img, HD_GFX_IMG_BUF *dst_img, HD_GFX_SCALE_QUALITY quality) {
	HD_RESULT ret = HD_OK;
	HD_GFX_SCALE param;
	memset(&param, 0, sizeof(HD_GFX_SCALE));

	param.src_img.dim.w = src_img->dim.w;
	param.src_img.dim.h = src_img->dim.h;
	param.src_img.format = src_img->format;
	param.src_img.p_phy_addr[0] = src_img->p_phy_addr[0];
	param.src_img.lineoffset[0] = src_img->lineoffset[0];

	param.dst_img.dim.w = dst_img->dim.w;
	param.dst_img.dim.h = dst_img->dim.h;
	param.dst_img.format = dst_img->format;
	param.dst_img.p_phy_addr[0] = dst_img->p_phy_addr[0];
	param.dst_img.lineoffset[0] = dst_img->lineoffset[0];

	param.src_region.x = 0;
	param.src_region.y = 0;
	param.src_region.w = src_img->dim.w;
	param.src_region.h = src_img->dim.h;

	param.dst_region.x = 0;
	param.dst_region.y = 0;
	param.dst_region.w = dst_img->dim.w;
	param.dst_region.h = dst_img->dim.h;

	param.quality = quality;

	ret = hd_gfx_scale(&param);
	if (ret != HD_OK) {
		printf("hd_gfx_scale fail(%d)\n", ret);
		return ret;
	}
	
	return HD_OK;
}

HD_RESULT load_jpeg_and_preprocess(const char *filename, MEM_PARM *out_buffer) {
	HD_RESULT ret = HD_OK;
	if (out_buffer->size < 28 * 28) {
		printf("Error: output buffer is too small.\n");
		return HD_ERR_NG;
	}

	int width, height, channels;
	unsigned char *img_data = stbi_load(filename, &width, &height, &channels, 1); // force 1 channel (grayscale)
        if (!img_data) {
		printf("Error: could not load image %s\n", filename);
		return HD_ERR_NG;
	}

	//alloc some hdal memory
	MEM_PARM orig_img = {0};
	ret = mem_alloc(&orig_img, "orig_img", (UINT32)(width * height));
	if (ret != HD_OK) {
		printf("Error: could not allocate memory for argb image\n");
		stbi_image_free(img_data);
		return ret;
	}

	// Copy the image data to the allocated memory(va=virtual address, pa=physical address)
	memcpy((void *)orig_img.va, (void *)img_data, width * height);
	hd_common_mem_flush_cache((void *)orig_img.va, orig_img.size);

	HD_GFX_IMG_BUF src_img = {0}, dst_img = {0};

	/* src_img*/
	src_img.dim.w             = width;
	src_img.dim.h             = height;
	src_img.format            = HD_VIDEO_PXLFMT_Y8;
	src_img.p_phy_addr[0]     = orig_img.pa;
	src_img.lineoffset[0]     = width;

	/* dst_img*/
	dst_img.dim.w             = 28;
	dst_img.dim.h             = 28;
	dst_img.format            = HD_VIDEO_PXLFMT_Y8;
	dst_img.p_phy_addr[0]     = out_buffer->pa;
	dst_img.lineoffset[0]     = 28;

	ret = scale_image(&src_img, &dst_img, HD_GFX_SCALE_QUALITY_INTEGRATION);
	if (ret != HD_OK) {
		printf("Error: could not convert image to grayscale and scale\n");
		stbi_image_free(img_data);
		mem_free(&orig_img);
		return ret;
	}

	stbi_image_free(img_data);
	mem_free(&orig_img);

	return ret;
}

static HD_RESULT input_init(void)
{
	HD_RESULT ret = HD_OK;
	int  i;

	for (i = 0; i < 16; i++) {
		NET_IN *p_net = g_in + i;
		p_net->in_id = i;
	}
	return ret;
}

static HD_RESULT input_uninit(void)
{
	HD_RESULT ret = HD_OK;
	return ret;
}


static HD_RESULT input_set_config(NET_PATH_ID in_path, NET_IN_CONFIG *p_in_cfg)
{
	HD_RESULT ret = HD_OK;
	NET_IN *p_net = g_in + in_path;
	UINT32 in_id = p_net->in_id;

	memcpy((void *)&p_net->in_cfg, (void *)p_in_cfg, sizeof(NET_IN_CONFIG));
	printf("in_path(%u) in_id(%u) set in_cfg: file(%s), buf=(%u,%u,%u,%u,%08x)\r\n",
		in_path,
		in_id,
		p_net->in_cfg.input_filename,
		p_net->in_cfg.w,
		p_net->in_cfg.h,
		p_net->in_cfg.c,
		p_net->in_cfg.loff,
		p_net->in_cfg.fmt);

	return ret;
}

static HD_RESULT input_open(NET_PATH_ID in_path)
{
	HD_RESULT ret = HD_OK;
	NET_IN *p_net = g_in + in_path;
	UINT32 in_id = p_net->in_id;
	CHAR mem_name[23];
	snprintf(mem_name, 23, "ai_in_buf %u", in_id);

	ret = mem_alloc(&p_net->input_mem, mem_name, 28*28);
	if (ret != HD_OK) {
		printf("in_path(%lu) in_id(%u) alloc ai_in_buf fail\r\n", in_path, in_id);
		return HD_ERR_FAIL;
	}
  
  //
	if (load_jpeg_and_preprocess(p_net->in_cfg.input_filename, &p_net->input_mem) < 0) {
		printf("in_path(%lu) in_id(%u) load input buf(%s) fail\r\n", in_path, in_id, p_net->in_cfg.input_filename);
		mem_free(&p_net->input_mem);
		return HD_ERR_FAIL;
	}

	if (hd_common_mem_flush_cache((VOID *)p_net->input_mem.va, p_net->input_mem.size) != HD_OK) {
		printf("in_path(%lu) in_id(%u) flush cache fail\r\n", in_path, in_id);
		mem_free(&p_net->input_mem);
		return HD_ERR_FAIL;
	}
  
  //setup src_img
	p_net->src_img.width = p_net->in_cfg.w;
	p_net->src_img.height = p_net->in_cfg.h;
	p_net->src_img.channel = p_net->in_cfg.c;
	p_net->src_img.line_ofs = p_net->in_cfg.loff;
	p_net->src_img.fmt = p_net->in_cfg.fmt;
	p_net->src_img.pa   = p_net->input_mem.pa;
	p_net->src_img.va   = p_net->input_mem.va;
	p_net->src_img.sign = MAKEFOURCC('A', 'B', 'U', 'F');
	p_net->src_img.size = p_net->input_mem.size;

	return ret;
}

static HD_RESULT input_close(NET_PATH_ID in_path)
{
	HD_RESULT ret = HD_OK;
	NET_IN *p_net = g_in + in_path;

	mem_free(&p_net->input_mem);

	return ret;
}

static HD_RESULT input_start(NET_PATH_ID in_path)
{
	HD_RESULT ret = HD_OK;
	return ret;
}

static HD_RESULT input_stop(NET_PATH_ID in_path)
{
	HD_RESULT ret = HD_OK;
	return ret;
}

static HD_RESULT input_pull_buf(NET_PATH_ID in_path, VENDOR_AI3_BUF *p_in, INT32 wait_ms)
{
	HD_RESULT ret = HD_OK;
	NET_IN *p_net = g_in + in_path;

	memcpy((void *)p_in, (void *)&(p_net->src_img), sizeof(VENDOR_AI3_BUF));
	return ret;
}
///////////////////////////////////////////////////////////////////////////////

/*-----------------------------------------------------------------------------*/
/* Network Functions                                                             */
/*-----------------------------------------------------------------------------*/

typedef struct _NET_PROC_CONFIG {

	CHAR model_filename[256];
	INT32 binsize;
	void *p_share_model;

	CHAR label_filename[256];

} NET_PROC_CONFIG;

typedef struct _NET_PROC {

	NET_PROC_CONFIG net_cfg;
	MEM_PARM proc_mem;
	UINT32 proc_id;

	CHAR out_class_labels[MAX_CLASS_NUMBER * LABEL_LEN];
	MEM_PARM rslt_mem;
	MEM_PARM io_mem;
	MEM_PARM intl_mem;
	MEM_PARM *out_mem;
	VENDOR_AI3_NET_INFO net_info;

} NET_PROC;

static NET_PROC g_net[16] = {0};

static INT32 _getsize_model(char *filename)
{
	FILE *bin_fd;
	UINT32 bin_size = 0;

	bin_fd = fopen(filename, "rb");
	if (!bin_fd) {
		printf("get bin(%s) size fail\n", filename);
		return (-1);
	}

	fseek(bin_fd, 0, SEEK_END);
	bin_size = ftell(bin_fd);
	fseek(bin_fd, 0, SEEK_SET);
	fclose(bin_fd);

	return bin_size;
}

static UINT32 _load_model(CHAR *filename, UINTPTR va)
{
	FILE  *fd;
	UINT32 file_size = 0, read_size = 0;
	const UINTPTR model_addr = va;
	//DBG_DUMP("model addr = %#lx\r\n", model_addr);

	fd = fopen(filename, "rb");
	if (!fd) {
		printf("load model(%s) fail\r\n", filename);
		return 0;
	}

	fseek(fd, 0, SEEK_END);
	file_size = ALIGN_CEIL_4(ftell(fd));
	fseek(fd, 0, SEEK_SET);

	read_size = fread((void *)model_addr, 1, file_size, fd);
	if (read_size != file_size) {
		printf("size mismatch, real = %d, idea = %d\r\n", (int)read_size, (int)file_size);
	}
	fclose(fd);

	printf("load model(%s) ok\r\n", filename);
	return read_size;
}

static HD_RESULT network_init(void)
{
	HD_RESULT ret = HD_OK;

	// call init
	{
		VENDOR_AI3_DEV_CFG dev_cfg = {0};

		ret = vendor_ai3_dev_init(&dev_cfg);
		if (ret != HD_OK) {
			printf("vendor_ai3_dev_init fail=%d\n", ret);
			return ret;
		}
	}
	// dump AI3 version
	{
		VENDOR_AI3_VER ai3_ver = {0};
		ret = vendor_ai3_dev_get(VENDOR_AI3_CFG_VER, &ai3_ver);
		if (ret != HD_OK) {
			printf("vendor_ai3_dev_get(CFG_VER) fail=%d\n", ret);
			return ret;
		}
		printf("vendor_ai version = %s\r\n", ai3_ver.vendor_ai_impl_version);
		printf("kflow_ai  version = %s\r\n", ai3_ver.kflow_ai_impl_version);
		printf("kdrv_ai   version = %s\r\n", ai3_ver.kdrv_ai_impl_version);
	}
	return ret;
}

static HD_RESULT network_uninit(void)
{
	HD_RESULT ret = HD_OK;

	ret = vendor_ai3_dev_uninit();
	if (ret != HD_OK) {
		printf("vendor_ai3_dev_uninit fail=%d\n", ret);
	}
	return ret;
}

INT32 network_mem_config(NET_PATH_ID net_path, HD_COMMON_MEM_INIT_CONFIG *p_mem_cfg, void *p_cfg)
{
	NET_PROC *p_net = g_net + net_path;
	NET_PROC_CONFIG *p_proc_cfg = (NET_PROC_CONFIG *)p_cfg;
#if defined(_NVT_NVR_SDK_)
	int sys_fd;
	struct nvtmem_hdal_base sys_hdal;
	uintptr_t hdal_start_addr0, hdal_start_addr1;

	sys_fd = open("/dev/nvtmem0", O_RDWR);
	if (sys_fd < 0) {
		printf("Error: cannot open /dev/nvtmem0 device.\n");
		exit(0);
	}
	if (ioctl(sys_fd, NVTMEM_GET_DTS_HDAL_BASE, &sys_hdal) < 0) {
		printf("PCIE_SYS_IOC_HDALBASE! \n");
		close(sys_fd);
		exit(0);
	}
	close(sys_fd);

	/* init ddr0 user_blk */
	hdal_start_addr0 = sys_hdal.base[0];
	p_mem_cfg->pool_info[0].start_addr = hdal_start_addr0;
	p_mem_cfg->pool_info[0].blk_cnt = 1;
	p_mem_cfg->pool_info[0].blk_size = 200 * 1024 * 1024;
	p_mem_cfg->pool_info[0].type = HD_COMMON_MEM_USER_BLK;
	p_mem_cfg->pool_info[0].ddr_id = sys_hdal.ddr_id[0];
	printf("create ddr%d: hdal_memory(%#lx, %ldKB), usr_blk(%#lx, %dKB)\n", p_mem_cfg->pool_info[0].ddr_id,
			hdal_start_addr0, sys_hdal.size[0] / 1024, p_mem_cfg->pool_info[0].start_addr,
			p_mem_cfg->pool_info[0].blk_size * p_mem_cfg->pool_info[0].blk_cnt / 1024);

	/* init ddr1 user_blk, if ddr1 is exist */
	if (sys_hdal.size[1] != 0) {
		hdal_start_addr1 = sys_hdal.base[1];
		p_mem_cfg->pool_info[1].start_addr = hdal_start_addr1;
		p_mem_cfg->pool_info[1].blk_cnt = 1;
		p_mem_cfg->pool_info[1].blk_size = 200 * 1024 * 1024;
		p_mem_cfg->pool_info[1].type = HD_COMMON_MEM_USER_BLK;
		p_mem_cfg->pool_info[1].ddr_id = sys_hdal.ddr_id[1];
		printf("create ddr%d: hdal_memory(%#lx, %ldKB) usr_blk(%#lx, %dKB)\n", p_mem_cfg->pool_info[1].ddr_id,
				hdal_start_addr1, sys_hdal.size[1] / 1024, p_mem_cfg->pool_info[1].start_addr,
				p_mem_cfg->pool_info[1].blk_size * p_mem_cfg->pool_info[1].blk_cnt / 1024);
	} else {
		printf("create ddr1: hdal_memory(%#lx, %ldKB) is not exist\n", sys_hdal.base[1], sys_hdal.size[1] / 1024);
	}
	usleep(30000); // wait for printf completely
#endif

	memcpy((void *)&p_net->net_cfg, (void *)p_proc_cfg, sizeof(NET_PROC_CONFIG));
	if (strlen(p_net->net_cfg.model_filename) == 0) {
		printf("net_path(%u) input model is null\r\n", net_path);
		return HD_ERR_NG;
	}

	p_net->net_cfg.binsize = _getsize_model(p_net->net_cfg.model_filename);
	if (p_net->net_cfg.binsize <= 0) {
		printf("net_path(%u) input model is not exist?\r\n", net_path);
		return HD_ERR_NG;
	}

	printf("net_path(%u) set net_mem_cfg: model-file(%s), binsize=%d\r\n",
		net_path,
		p_net->net_cfg.model_filename,
		p_net->net_cfg.binsize);

	printf("net_path(%u) set net_mem_cfg: label-file(%s)\r\n",
		net_path,
		p_net->net_cfg.label_filename);

	return HD_OK;
}

static HD_RESULT network_set_config(NET_PATH_ID net_path, NET_PROC_CONFIG *p_proc_cfg)
{
	HD_RESULT ret = HD_OK;
	// nothing to set
	return ret;
}

static HD_RESULT network_alloc_io_buf(NET_PATH_ID net_path, UINT32 req_size)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + net_path;
	CHAR mem_name[23] ;
	snprintf(mem_name, 23, "ai_io_buf %u", net_path);

	ret = mem_alloc(&p_net->io_mem, mem_name, req_size);
	if (ret != HD_OK) {
		printf("net_path(%lu) alloc ai_io_buf fail\r\n", net_path);
		return HD_ERR_FAIL;
	}

	printf("alloc_io_buf: work buf, pa = %#lx, va = %#lx, size = %lu\r\n", p_net->io_mem.pa, p_net->io_mem.va, p_net->io_mem.size);

	return ret;
}

static HD_RESULT network_free_io_buf(NET_PATH_ID net_path)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + net_path;

	if (p_net->io_mem.pa && p_net->io_mem.va) {
		mem_free(&p_net->io_mem);
	}
	return ret;
}

static HD_RESULT network_alloc_intl_buf(NET_PATH_ID net_path, UINT32 req_size)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + net_path;

	CHAR mem_name[23] ;
	snprintf(mem_name, 23, "ai_ronly_buf %u", net_path);

	ret = mem_alloc(&p_net->intl_mem, mem_name, req_size);
	if (ret != HD_OK) {
		printf("net_path(%lu) alloc ai_ronly_buf fail\r\n", net_path);
		return HD_ERR_FAIL;
	}

	printf("alloc_intl_buf: internal buf, pa = %#lx, va = %#lx, size = %lu\r\n", p_net->intl_mem.pa, p_net->intl_mem.va, p_net->intl_mem.size);

	return ret;
}

static HD_RESULT network_free_intl_buf(NET_PATH_ID net_path)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + net_path;

	if (p_net->intl_mem.pa && p_net->intl_mem.va) {
		mem_free(&p_net->intl_mem);
	}
	return ret;
}

#if DBG_IO_BUF_INFO_DUMP
static HD_RESULT network_io_buf_info_dump_print_info(VENDOR_AI3_BUF_INFO *buf_info)
{
	printf("  +- name    = %s\r\n", buf_info->name);
	printf("  +- fmt     = 0x%08x\r\n", (unsigned int)buf_info->fmt);
	printf("    +- bits  = %d\r\n", HD_VIDEO_PXLFMT_BITS(buf_info->fmt));
	printf("    +- sign  = %d\r\n", HD_VIDEO_PXLFMT_SIGN(buf_info->fmt));
	printf("    +- int   = %d\r\n", HD_VIDEO_PXLFMT_INT(buf_info->fmt));
	printf("    +- frac  = %d\r\n", HD_VIDEO_PXLFMT_FRAC(buf_info->fmt));
	printf("  +- width   = %d\r\n", (int)buf_info->width);
	printf("  +- height  = %d\r\n", (int)buf_info->height);
	printf("  +- channel = %d\r\n", (int)buf_info->channel);
	printf("  +- batch   = %d\r\n", (int)buf_info->batch_num);
	printf("  +- time    = %d\r\n", (int)buf_info->time);
	printf("  +- layout  = %s\r\n", buf_info->layout);
	printf("\r\n");
	return HD_OK;
}

static HD_RESULT network_io_buf_info_dump(NET_PATH_ID net_path, VENDOR_AI3_CFG_BUF *model_buf)
{
	HD_RESULT ret = HD_OK;
	VENDOR_AI3_IO_BUF_INFO ai_io_info = {0};

	ai_io_info.model_buf.pa   = model_buf->pa;
	ai_io_info.model_buf.va   = model_buf->va;
	ai_io_info.model_buf.size = model_buf->size;

	// query in/out buffer count
	ret = vendor_ai3_dev_get(VENDOR_AI3_CFG_IO_CNT, &ai_io_info); // get in_buf_cnt & out_buf_cnt
	if (ret != HD_OK) {
		printf("net_path(%u) vendor_ai3_dev_get(VENDOR_AI3_CFG_IO_CNT) fail=%d\n", net_path, ret);
		return HD_ERR_FAIL;
	}
	printf("net_path(%u) in_buf_cnt = %d, out_buf_cnt = %d\r\n", net_path, (int)ai_io_info.in_buf_cnt, (int)ai_io_info.out_buf_cnt);

	// alloc context for in/out buffer info
	ai_io_info.in_buf_info  = (VENDOR_AI3_BUF_INFO *)malloc(sizeof(VENDOR_AI3_BUF_INFO) * ai_io_info.in_buf_cnt);
	ai_io_info.out_buf_info = (VENDOR_AI3_BUF_INFO *)malloc(sizeof(VENDOR_AI3_BUF_INFO) * ai_io_info.out_buf_cnt);
	if (ai_io_info.in_buf_info == NULL || ai_io_info.out_buf_info == NULL) {
		printf("net_path(%u) malloc in_buf_info/out_buf_info failed...\n", net_path);
		ret = HD_ERR_FAIL;
		goto free_context;
	}

	// query in/out buffer info
	vendor_ai3_dev_get(VENDOR_AI3_CFG_IO_INFO, &ai_io_info); // get in_buf_info & out_buf_info
	if (ret != HD_OK) {
		printf("net_path(%u) vendor_ai3_dev_get(VENDOR_AI3_CFG_IO_INFO) fail=%d\n", net_path, ret);
		ret = HD_ERR_FAIL;
		goto free_context;
	}

	// print in/out buffer info
	{
		UINT32 i = 0;
		// in
		for (i = 0; i < ai_io_info.in_buf_cnt ; i++) {
			printf("input [%d] =>\r\n", (int)i);
			network_io_buf_info_dump_print_info(&ai_io_info.in_buf_info[i]);
		}
		// out
		for (i = 0; i < ai_io_info.out_buf_cnt ; i++) {
			printf("output [%d] =>\r\n", (int)i);
			network_io_buf_info_dump_print_info(&ai_io_info.out_buf_info[i]);
		}
	}

free_context:
	// free context
	if (ai_io_info.in_buf_info) {
		free(ai_io_info.in_buf_info);
	}
	if (ai_io_info.out_buf_info) {
		free(ai_io_info.out_buf_info);
	}

	return ret;
}
#endif // DBG_IO_BUF_INFO_DUMP
#if DBG_MODEL_VERSION
static HD_RESULT network_version_dump(NET_PATH_ID net_path, VENDOR_AI3_CFG_BUF *model_buf){
	HD_RESULT ret = HD_OK;
	VENDOR_AI3_TOOL_VER ai_model_ver = {0};

	ai_model_ver.model_buf.pa   = model_buf->pa;
	ai_model_ver.model_buf.va   = model_buf->va;
	ai_model_ver.model_buf.size = model_buf->size;

	ret = vendor_ai3_dev_get(VENDOR_AI3_CFG_TOOLVER, &ai_model_ver); // get model version
	if (ret != HD_OK) {
		printf("net_path(%u) vendor_ai3_dev_get(VENDOR_AI3_CFG_TOOLVER) fail=%d\n", net_path, ret);
		return HD_ERR_FAIL;
	}
	printf("net_path(%u) tool gen version = %s\r\n", net_path, ai_model_ver.tool_version);

	return ret;
}
#endif // DBG_MODEL_VERSION
static HD_RESULT network_open(NET_PATH_ID net_path)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + net_path;
	UINT32 loadsize = 0;
	CHAR mem_name[23] ;

	snprintf(mem_name, 23, "model.bin %u", net_path);

	if (strlen(p_net->net_cfg.model_filename) == 0) {
		printf("net_path(%u) input model is null\r\n", net_path);
		return 0;
	}
	ret  =  mem_alloc(&p_net->proc_mem, mem_name,  p_net->net_cfg.binsize);
	if (ret != HD_OK) {
		printf("net_path(%u) mem_alloc model.bin fail=%d\n", net_path, ret);
		return HD_ERR_FAIL;
	}
	//load file
	loadsize = _load_model(p_net->net_cfg.model_filename, p_net->proc_mem.va);

	if (loadsize <= 0) {
		printf("net_path(%u) input model load fail: %s\r\n", net_path, p_net->net_cfg.model_filename);
		return 0;
	}

#if DBG_MODEL_VERSION
	// dump model version before open()
	ret = network_version_dump(net_path, (VENDOR_AI3_CFG_BUF *)&p_net->proc_mem);
	if (ret != HD_OK) {
		printf("net_path(%u) network_version_dump() fail=%d\n", net_path, ret);
		return HD_ERR_FAIL;
	}
#endif

#if DBG_IO_BUF_INFO_DUMP
	// query model in/out buffer info before open()
	ret = network_io_buf_info_dump(net_path, (VENDOR_AI3_CFG_BUF *)&p_net->proc_mem);
	if (ret != HD_OK) {
		printf("net_path(%u) network_io_buf_info_dump() fail=%d\n", net_path, ret);
		return HD_ERR_FAIL;
	}
#endif

	// query model info for WORKBUF/RONLYBUF size , then alloc WORKBUF/RONLYBUF
	{
		VENDOR_AI3_MODEL_INFO model_info = {0};

		model_info.model_buf.pa   = p_net->proc_mem.pa;
		model_info.model_buf.va   = p_net->proc_mem.va;
		model_info.model_buf.size = p_net->proc_mem.size;
#if DBG_OUT_DUMP
		model_info.ctrl |= (CTRL_BUF_DEBUG | CTRL_JOB_DEBUG | CTRL_JOB_DUMPOUT);
#endif
#if DBG_TIME_DUMP
		model_info.ctrl |= (CTRL_JOB_DEBUG | CTRL_JOB_PERFTIME);
#endif
#if DBG_BW_DUMP
		model_info.ctrl |= (CTRL_JOB_DEBUG | CTRL_JOB_PERFBW);
#endif
#if DBG_PERF_DUMP
		model_info.ctrl |= (CTRL_JOB_DEBUG | CTRL_JOB_PERFTIME | CTRL_JOB_PERFBW);
#endif
		ret = vendor_ai3_dev_get(VENDOR_AI3_CFG_MODEL_INFO, &model_info);
		if (ret != HD_OK) {
			printf("net_path(%u) vendor_ai3_dev_get(MODEL_INFO) fail=%d\n", net_path, ret);
			return HD_ERR_FAIL;
		}

		printf("model_info get => workbuf size = %d, ronlybuf size = %d\r\n", model_info.proc_mem.buf[AI3_PROC_BUF_WORKBUF].size, model_info.proc_mem.buf[AI3_PROC_BUF_RONLYBUF].size);

		// alloc WORKBUF/RONLYBUF
		ret = network_alloc_intl_buf(net_path, model_info.proc_mem.buf[AI3_PROC_BUF_RONLYBUF].size);
		if (ret != HD_OK) {
			printf("net_path(%u) alloc ronlybuf fail=%d\n", net_path, ret);
			return HD_ERR_FAIL;
		}

		ret = network_alloc_io_buf(net_path, model_info.proc_mem.buf[AI3_PROC_BUF_WORKBUF].size);
		if (ret != HD_OK) {
			printf("net_path(%u) alloc workbuf fail=%d\n", net_path, ret);
			return HD_ERR_FAIL;
		}
	}

	// call open()
	{
		VENDOR_AI3_PROC_CFG proc_cfg = {0};

		proc_cfg.model_buf.pa   = p_net->proc_mem.pa;
		proc_cfg.model_buf.va   = p_net->proc_mem.va;
		proc_cfg.model_buf.size = p_net->proc_mem.size;

		proc_cfg.proc_mem.buf[AI3_PROC_BUF_RONLYBUF].pa   = p_net->intl_mem.pa;
		proc_cfg.proc_mem.buf[AI3_PROC_BUF_RONLYBUF].va   = p_net->intl_mem.va;
		proc_cfg.proc_mem.buf[AI3_PROC_BUF_RONLYBUF].size = p_net->intl_mem.size;

		proc_cfg.proc_mem.buf[AI3_PROC_BUF_WORKBUF].pa   = p_net->io_mem.pa;
		proc_cfg.proc_mem.buf[AI3_PROC_BUF_WORKBUF].va   = p_net->io_mem.va;
		proc_cfg.proc_mem.buf[AI3_PROC_BUF_WORKBUF].size = p_net->io_mem.size;

		proc_cfg.plugin[AI3_PLUGIN_CPU] = vendor_ai_cpu1_get_engine();
#if DBG_OUT_DUMP
		proc_cfg.ctrl |= (CTRL_BUF_DEBUG | CTRL_JOB_DEBUG | CTRL_JOB_DUMPOUT);
#endif
#if DBG_TIME_DUMP
		proc_cfg.ctrl |= (CTRL_JOB_DEBUG | CTRL_JOB_PERFTIME);
#endif
#if DBG_BW_DUMP
		proc_cfg.ctrl |= (CTRL_JOB_DEBUG | CTRL_JOB_PERFBW);
#endif
#if DBG_PERF_DUMP
		proc_cfg.ctrl |= (CTRL_JOB_DEBUG | CTRL_JOB_PERFTIME | CTRL_JOB_PERFBW);
#endif
		ret = vendor_ai3_net_open(&p_net->proc_id, &proc_cfg, &p_net->net_info);
		if (ret != HD_OK) {
			printf("net_path(%u) vendor_ai3_net_open() fail=%d\n", net_path, ret);
			return HD_ERR_FAIL;
		} else {
			printf("net_path(%u) open success => get proc_id(%u)\r\n", net_path, p_net->proc_id);
		}
	}
	return ret;
}

static HD_RESULT network_close(NET_PATH_ID net_path)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + net_path;
	UINT32 proc_id = p_net->proc_id;
	UINT32 i;

	// close
	ret = vendor_ai3_net_close(proc_id);
	if (ret != HD_OK) {
		printf("net_path(%u), proc_id(%u) vendor_ai3_net_close fail=%d\n", net_path, proc_id, ret);
		return HD_ERR_FAIL;
	}

	if ((ret = network_free_intl_buf(net_path)) != HD_OK) {
		return ret;
	}

	if ((ret = network_free_io_buf(net_path)) != HD_OK) {
		return ret;
	}

	mem_free(&p_net->proc_mem);

	for (i = 0 ; i < p_net->net_info.out_buf_cnt; i++) {
		if(p_net->out_mem && p_net->out_mem[i].va)
			mem_free(&p_net->out_mem[i]);
	}
	if (p_net->out_mem) {
		free(p_net->out_mem);
	}

	memset(&p_net->net_info, 0, sizeof(VENDOR_AI3_NET_INFO));

	return ret;
}
///////////////////////////////////////////////////////////////////////////////

typedef struct _VIDEO_LIVEVIEW {

	// (1) input
	NET_IN_CONFIG net_in_cfg;
	NET_PATH_ID in_path;

	// (2) network
	NET_PROC_CONFIG net_proc_cfg;
	NET_PATH_ID net_path;
	pthread_t  proc_thread_id;
	UINT32 proc_start;
	UINT32 proc_exit;
	UINT32 proc_oneshot;
} VIDEO_LIVEVIEW;

static HD_RESULT init_module(void)
{
	HD_RESULT ret;
	if ((ret = hd_gfx_init()) != HD_OK) {
		return ret;
	}
	if ((ret = input_init()) != HD_OK) {
		return ret;
	}
	if ((ret = network_init()) != HD_OK) {
		return ret;
	}
	return HD_OK;
}

static HD_RESULT open_module(VIDEO_LIVEVIEW *p_stream)
{
	HD_RESULT ret;
	if ((ret = input_open(p_stream->in_path)) != HD_OK) {
		return ret;
	}
	if ((ret = network_open(p_stream->net_path)) != HD_OK) {
		return ret;
	}
	return HD_OK;
}

static HD_RESULT close_module(VIDEO_LIVEVIEW *p_stream)
{
	HD_RESULT ret;
	if ((ret = input_close(p_stream->in_path)) != HD_OK) {
		return ret;
	}
	if ((ret = network_close(p_stream->net_path)) != HD_OK) {
		return ret;
	}
	return HD_OK;
}

static HD_RESULT exit_module(void)
{
	HD_RESULT ret;
	if ((ret = hd_gfx_uninit()) != HD_OK) {
		return ret;
	}
	if ((ret = input_uninit()) != HD_OK) {
		return ret;
	}
	if ((ret = network_uninit()) != HD_OK) {
		return ret;
	}
	return HD_OK;
}

uintptr_t get_post_buf(uint32_t size)
{
	uintptr_t buf = (uintptr_t)malloc(size);

	return buf;

}

VOID release_post_buf(VOID *ptr)
{
	if (ptr) {
		free(ptr);
	}
	return;
}
///////////////////////////////////////////////////////////////////////////////

static VOID *network_user_thread(VOID *arg);

#if DUMP_POSTPROC_INFO
static HD_RESULT network_dump_out_buf(NET_PATH_ID net_path, VENDOR_AI3_BUF *p_outbuf)
{
    HD_RESULT ret = HD_OK;
	// 0-9
    INT32 size = 10;

    FLOAT *p_scores = (FLOAT *)get_post_buf(size * sizeof(FLOAT));
    ret = vendor_ai_cpu_util_fixed2float((VOID *)p_outbuf->va, p_outbuf->fmt, p_scores, p_outbuf->scale_ratio, size, p_outbuf->zero_point);

    // argmax
    FLOAT max_score = -1.0f;
    INT32 max_index = -1;

    for (int i = 0; i < size; i++) {
        if (p_scores[i] > max_score) {
            max_score = p_scores[i];
            max_index = i;
        }
    }

    printf("=======================================\n");
    printf("Prediction Result: %d\n", max_index);
    printf("Confidence: %f\n", max_score);
    printf("=======================================\n");

    release_post_buf(p_scores);
    return ret;
}
#endif

static HD_RESULT set_buf_by_out_path_list(NET_PATH_ID net_path)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + net_path;
	UINT32 proc_id = p_net->proc_id;
	UINT32 i;
	VENDOR_AI3_BUF ai_buf = {0};

	/* get out buf */
	for (i = 0; i < p_net->net_info.out_buf_cnt; i++) {
		// get out buf (by out path list)
		ret = vendor_ai3_net_get(proc_id, p_net->net_info.out_path_list[i], &ai_buf);
		if (HD_OK != ret) {
			printf("proc_id(%u) get out buf fail, i(%d), out_path(0x%lx)\n", proc_id, i, p_net->net_info.out_path_list[i]);
			goto exit;
		}
		if (ai_buf.size > p_net->out_mem[i].size) {
			printf("output size %u < ai_buf.size %u\r\n", p_net->out_mem[i].size, ai_buf.size);
			goto exit;
		}

		ai_buf.va = p_net->out_mem[i].va;
		ai_buf.pa = p_net->out_mem[i].pa;
		ai_buf.size = p_net->out_mem[i].size;

		ret = vendor_ai3_net_set(proc_id, p_net->net_info.out_path_list[i], &ai_buf);
		if (HD_OK != ret) {
			printf("proc_id(%u)set output buf fail !! (%lu)\n", proc_id, i);
			goto exit;
		}
	}

exit:
	return ret;
}

static HD_RESULT get_buf_by_out_path_list(NET_PATH_ID net_path)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + net_path;
	UINT32 proc_id = p_net->proc_id;
	UINT32 i;
	VENDOR_AI3_BUF ai_buf = {0};

	/* get out buf */
	for (i = 0; i < p_net->net_info.out_buf_cnt; i++) {
		// get out buf (by out path list)
		ret = vendor_ai3_net_get(proc_id, p_net->net_info.out_path_list[i], &ai_buf);
		if (HD_OK != ret) {
			printf("net_path(%u), proc_id(%u) get out buf fail, i(%d), out_path(0x%lx)\n", net_path, proc_id, i,  p_net->net_info.out_path_list[i]);
			goto exit;
		}
#if DUMP_POSTPROC_INFO
		// dump out buf
		printf("dump_out_buf: path_id: 0x%lx\n", p_net->net_info.out_path_list[i]);
        if (hd_common_mem_flush_cache((VOID *)ai_buf.va, ai_buf.size) != HD_OK) {
            printf("flush cache failed.\r\n");
        }
		ret = network_dump_out_buf(net_path, &ai_buf);
		if (HD_OK != ret) {
			printf("net_path(%u) dump out buf fail !!\n", net_path);
			goto exit;
		}
#endif
	}

exit:
	return ret;
}

static HD_RESULT set_buf_by_in_path_list(VIDEO_LIVEVIEW *p_stream)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + p_stream->net_path;
	UINT32 proc_id = p_net->proc_id;
	UINT32 i;
	VENDOR_AI3_BUF ai_buf = {0};

	/* get out buf */
	for (i = 0; i < p_net->net_info.in_buf_cnt; i++) {
		// get out buf (by out path list)
		ret = vendor_ai3_net_get(proc_id, p_net->net_info.in_path_list[i], &ai_buf);
		if (HD_OK != ret) {
			printf("net_path(%u), proc_id(%u) get in buf fail, i(%d), in_path(0x%lx)\n", p_stream->net_path, proc_id, i,  p_net->net_info.in_path_list[i]);
			goto exit;
		}

		ret = input_pull_buf((p_stream->in_path + i), &ai_buf, 0);
		if (HD_OK != ret) {
			printf("in_path(%u) pull input fail !!\n", (p_stream->in_path + i));
			goto exit;
		}
#if YUV_CROP
		VENDOR_AI3_BUF y_buf = {0};
		VENDOR_AI3_BUF uv_buf = {0};
		//  CORP img 400*200 from original YUV input (512*376)
		//  start (x,y) is (64, 100) 
		//  apply ai3_buf list 

		y_buf.width = 400;
		y_buf.height = 200;
		y_buf.channel = 1;
		y_buf.line_ofs = 512;
		y_buf.fmt = HD_VIDEO_PXLFMT_Y8;
		y_buf.sign = MAKEFOURCC('A','B','U','F');
		y_buf.size = 512*376;
		y_buf.va = ai_buf.va + 64 + 100 * 512;
		y_buf.pa = ai_buf.pa + 64 + 100 * 512;
		y_buf.p_next = &uv_buf;
		
		uv_buf.width = 400;
		uv_buf.height = 200;
		uv_buf.channel = 1;
		uv_buf.line_ofs = 512;
		uv_buf.fmt = HD_VIDEO_PXLFMT_UV;
		uv_buf.sign = MAKEFOURCC('A','B','U','F');
		uv_buf.size = 512*376;
		uv_buf.va = ai_buf.va + 512 * 376 + 64 + 100/2 * 512;
		uv_buf.pa = ai_buf.pa + 512 * 376 + 64 + 100/2 * 512;
		ret = vendor_ai3_net_set(proc_id, p_net->net_info.in_path_list[i], &y_buf);
#else 
		ret = vendor_ai3_net_set(proc_id, p_net->net_info.in_path_list[i], &ai_buf);
#endif
		if (HD_OK != ret) {
			printf("proc_id(%u)push input fail !! i(%lu)\n", proc_id, i);
			goto exit;
		}
	}

exit:
	return ret;
}

static HD_RESULT allocate_buf_by_out_path_list(NET_PATH_ID net_path)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + net_path;
	UINT32 proc_id = p_net->proc_id;
	UINT32 i;
	VENDOR_AI3_BUF ai_buf = {0};
	CHAR mem_name[23];

	/* get out path list */
	p_net->out_mem = (MEM_PARM *)malloc(sizeof(MEM_PARM) * p_net->net_info.out_buf_cnt);
	memset(p_net->out_mem, 0, sizeof(MEM_PARM) * p_net->net_info.out_buf_cnt);

	/* get out buf */
	for (i = 0; i < p_net->net_info.out_buf_cnt; i++) {
		// get out buf (by out path list)
		ret = vendor_ai3_net_get(proc_id, p_net->net_info.out_path_list[i], &ai_buf);
		if (HD_OK != ret) {
			printf("proc_id(%u) get out buf fail, i(%d), out_path(0x%lx)\n", proc_id, i, p_net->net_info.out_path_list[i]);
			goto exit;
		}

		// // allocate in buf
		snprintf(mem_name, 23, "output_buf %u", i);
		ret = mem_alloc(&p_net->out_mem[i], mem_name,  ai_buf.size);
		if (ret != HD_OK) {
			printf("proc_id(%lu) alloc ai_in_buf fail\r\n", proc_id);
			goto exit;
		}
		
		printf("alloc_outbuf: pa = 0x%lx, va = 0x%lx, size = %u\n", p_net->out_mem[i].pa, p_net->out_mem[i].va, p_net->out_mem[i].size);
	}
	return ret;
exit:
	for (i = 0; i < p_net->net_info.out_buf_cnt; i++) {
		if(p_net->out_mem && p_net->out_mem[i].va) {
			mem_free(&p_net->out_mem[i]);
		}
	}
	if (p_net->out_mem) {
		free(p_net->out_mem);
	}

	return ret;
}

#if (DBG_PERF_TIME_UT || DBG_PERF_TIME_UT2)
static HD_RESULT perf_begin(void)
{
	vendor_ai3_dev_perf_begin(VENDOR_AI3_PERF_ID_TIME_UT);
	return HD_OK;
}

static HD_RESULT perf_end(void)
{
	UINT32 i;
	VENDOR_AI3_PERF_TIME_UT perf_time_ut = {0};
	vendor_ai3_dev_perf_end(VENDOR_AI3_PERF_ID_TIME_UT, &perf_time_ut);
#if DBG_PERF_TIME_UT
	printf("\r\n ************* util-per-proc() *************\r\n");
#endif
#if DBG_PERF_TIME_UT2
	printf("\r\n ************* util-per-second *************\r\n");
#endif
	for (i=0; i<perf_time_ut.core_count; i++) {
	    printf("%8s: time(us) = %7d, util(%%) = %6.2f\r\n",
	    perf_time_ut.core[i].name, perf_time_ut.core[i].time, ((float)perf_time_ut.core[i].util)/100);
	}
	printf("\r\n");
	return HD_OK;
}
#endif

static HD_RESULT network_user_start(VIDEO_LIVEVIEW *p_stream)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + p_stream->net_path;
	UINT32 proc_id = p_net->proc_id;

	p_stream->proc_start = 0;
	p_stream->proc_exit = 0;
	p_stream->proc_oneshot = 0;

	ret = vendor_ai3_net_start(proc_id);
	if (HD_OK != ret) {
		printf("net_path(%u), proc_id(%u) vendor_ai3_net_start fail !!\n", p_stream->net_path, proc_id);
	}

	ret = pthread_create(&p_stream->proc_thread_id, NULL, network_user_thread, (VOID *)(p_stream));
	if (ret < 0) {
		return HD_ERR_FAIL;
	}

	p_stream->proc_start = 1;
	p_stream->proc_exit = 0;
	p_stream->proc_oneshot = 0;

	return ret;
}

static HD_RESULT network_user_oneshot(VIDEO_LIVEVIEW *p_stream)
{
	HD_RESULT ret = HD_OK;
	p_stream->proc_oneshot = 1;
	return ret;
}

static HD_RESULT network_user_stop(VIDEO_LIVEVIEW *p_stream)
{
	HD_RESULT ret = HD_OK;
	NET_PROC *p_net = g_net + p_stream->net_path;
	UINT32 proc_id = p_net->proc_id;

	p_stream->proc_exit = 1;

	if (p_stream->proc_thread_id) {
		pthread_join(p_stream->proc_thread_id, NULL);
	}

	//stop: should be call after last time proc
	ret = vendor_ai3_net_stop(proc_id);
	if (HD_OK != ret) {
		printf("net_path(%u), proc_id(%u) vendor_ai3_net_stop fail !!\n", p_stream->net_path, proc_id);
	}
	return ret;
}

static VOID *network_user_thread(VOID *arg)
{
	HD_RESULT ret = HD_OK;

	VIDEO_LIVEVIEW *p_stream = (VIDEO_LIVEVIEW *)arg;
	NET_PROC *p_net = g_net + p_stream->net_path;
	UINT32 proc_id = p_net->proc_id;
    UINT32 job_priority = VENDOR_AI_JOB_PRI(0);
    UINT32 core_mask = VENDOR_AI_CORE_MASK_DEFAULT;

	printf("\r\n");
	while (p_stream->proc_start == 0) {
		sleep(1);
	}

	printf("\r\n");
	while (p_stream->proc_exit == 0) {

		if (p_stream->proc_oneshot) {

#if (FREE_RUN==0)
			p_stream->proc_oneshot = 0;
#endif

			ret = set_buf_by_in_path_list(p_stream);
			if (HD_OK != ret) {
				printf("net_path(%u) get in_buf fail(%d) !!\n", p_stream->net_path, ret);
				goto skip;
			}

#if (FREE_RUN==0)
			printf("net_path(%u), proc_id(%u) oneshot ...\n", p_stream->net_path, proc_id);
#endif

			ret = set_buf_by_out_path_list(p_stream->net_path);
			if (HD_OK != ret) {
				printf("net_path(%u), proc_id(%u) set in_buf fail(%d) !!\n", p_stream->net_path, proc_id, ret);
				goto skip;
			}

            vendor_ai3_net_set(proc_id, VENDOR_AI3_NET_PARAM_JOB_PRI, &job_priority);

            vendor_ai3_net_set(proc_id, VENDOR_AI3_NET_PARAM_CORE_MASK, &core_mask);

#if DBG_PERF_TIME_UT
			perf_begin();
#endif
			// do net proc
			ret = vendor_ai3_net_proc(proc_id);
			if (HD_OK != ret) {
				printf("net_path(%u), proc_id(%u) proc fail !!\n", p_stream->net_path, proc_id);
				goto skip;
			}
#if DBG_PERF_TIME_UT
			perf_end();
#endif
#if (FREE_RUN==0)
			printf("net_path(%u), proc_id(%u) oneshot done!\n", p_stream->net_path, proc_id);
#endif

			// get buf by out_path_list
			ret = get_buf_by_out_path_list(p_stream->net_path);
			if (HD_OK != ret) {
				printf("net_path(%u) get out_buf fail(%d) !!\n", p_stream->net_path, ret);
				goto skip;
			}
		}
		usleep(20);
	}

skip:

	return 0;
}


#if DBG_PERF_TIME_UT2
typedef struct _PERF_MONITOR {

	pthread_t  proc_thread_id;
	UINT32 proc_start;
	UINT32 proc_exit;
} PERF_MONITOR;

static VOID *network_monitor_thread(VOID *arg)
{
	PERF_MONITOR *p_stream = (PERF_MONITOR *)arg;

	printf("\r\n");
	while (p_stream->proc_start == 0) {
		sleep(1);
	}

	printf("\r\n");
	while (p_stream->proc_exit == 0) {

		perf_begin();
		sleep(1); //sleep 1s
		perf_end();
		//usleep(100);
	}

	return 0;
}

static HD_RESULT network_monitor_start(PERF_MONITOR *p_stream)
{
	HD_RESULT ret = HD_OK;

	p_stream->proc_start = 0;
	p_stream->proc_exit = 0;

	ret = pthread_create(&p_stream->proc_thread_id, NULL, network_monitor_thread, (VOID *)(p_stream));
	if (ret < 0) {
		return HD_ERR_FAIL;
	}

	p_stream->proc_start = 1;
	p_stream->proc_exit = 0;

	return ret;
}

static HD_RESULT network_monitor_stop(PERF_MONITOR *p_stream)
{
	HD_RESULT ret = HD_OK;
	
	p_stream->proc_exit = 1;

	if (p_stream->proc_thread_id) {
		pthread_join(p_stream->proc_thread_id, NULL);
	}

	return ret;
}
#endif

/*-----------------------------------------------------------------------------*/
/* Interface Functions                                                         */
/*-----------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
	VIDEO_LIVEVIEW stream[1] = {0}; //0: net proc
#if DBG_PERF_TIME_UT2
	PERF_MONITOR mon[1] = {0};
#endif
	HD_COMMON_MEM_INIT_CONFIG mem_cfg = {0};
	HD_RESULT ret;
	INT key;

	if (argc != 2) {
		printf("usage : ai3_net <path_to_jpg_or_png>\n");
		return -1;
	}

	//net_in
	NET_IN_CONFIG in_cfg = {0};
	snprintf(in_cfg.input_filename, sizeof(in_cfg.input_filename), "%s", argv[1]);
	in_cfg.w = 28;
	in_cfg.h = 28;
	in_cfg.c = 1;
	in_cfg.loff = 28;
	in_cfg.fmt = HD_VIDEO_PXLFMT_Y8;

	//net proc
	NET_PROC_CONFIG net_cfg = {
		.model_filename = "para/nvt_model.bin",
	};

	printf("\r\n\r\n");

	stream[0].in_path = 0;
	stream[0].net_path = 0;

	// init hdal
	ret = hd_common_init(0);
	if (ret != HD_OK) {
		printf("hd_common_init fail=%d\n", ret);
		goto exit;
	}
//this is no longer need in 690
/*
#if defined(_BSP_NS02201_)
	// set project config for AI
	hd_common_sysconfig(0, (1<<16), 0, VENDOR_AI_CFG); //enable AI engine
#endif
*/

	// init mem
	{
		// config common pool
		network_mem_config(stream[0].net_path, &mem_cfg, &net_cfg);
	}
#if defined(_BSP_NS02201_)
	ret = hd_common_mem_init(&mem_cfg);
	if (HD_OK != ret) {
		printf("hd_common_mem_init err: %d\r\n", ret);
		goto exit;
	}
#endif

	// init all modules
	ret = init_module();
	if (ret != HD_OK) {
		printf("init fail=%d\n", ret);
		goto exit;
	}

	// set open config
	ret = input_set_config(stream[0].in_path, &in_cfg);
	if (HD_OK != ret) {
		printf("in_path(%u) input_set_config fail=%d\n", stream[0].in_path, ret);
		goto exit;
	}
	ret = network_set_config(stream[0].net_path, &net_cfg);
	if (HD_OK != ret) {
		printf("net_path(%u) network_set_config fail=%d\n", stream[0].net_path, ret);
		goto exit;
	}

	// open video_liveview modules
	ret = open_module(&stream[0]);
	if (ret != HD_OK) {
		printf("open fail=%d\n", ret);
		goto exit;
	}

	// start
	input_start(stream[0].in_path);
	network_user_start(&stream[0]);
	allocate_buf_by_out_path_list(stream[0].net_path);

#if DBG_PERF_TIME_UT2
	network_monitor_start(&mon[0]);
#endif
	printf("Enter q to quit\n");
	printf("Enter r to run once\n");
	do {
		key = getchar();
		if (key == 'r') {

			// run once
			network_user_oneshot(&stream[0]);
			continue;
		}

		if (key == 'q' || key == 0x3) {

			break;
		}
	} while (1);
#if DBG_PERF_TIME_UT2
	network_monitor_stop(&mon[0]);
#endif
	// stop
	input_stop(stream[0].in_path);
	network_user_stop(&stream[0]);

exit:

	// close video_liveview modules
	ret = close_module(&stream[0]);
	if (ret != HD_OK) {
		printf("close fail=%d\n", ret);
	}

	// uninit all modules
	ret = exit_module();
	if (ret != HD_OK) {
		printf("exit fail=%d\n", ret);
	}

#if defined(_BSP_NS02201_)
	// uninit memory
	ret = hd_common_mem_uninit();
	if (ret != HD_OK) {
		printf("mem fail=%d\n", ret);
	}
#endif
	// uninit hdal
	ret = hd_common_uninit();
	if (ret != HD_OK) {
		printf("common fail=%d\n", ret);
	}

	return ret;
}
