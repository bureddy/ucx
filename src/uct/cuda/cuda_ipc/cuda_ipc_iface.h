/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 *
 * Copyright (c) 2016-2017, NVIDIA Corporation.  All rights reserved.
 * See COPYRIGHT for license information
 */

#ifndef UCT_CUDA_IFACE_H
#define UCT_CUDA_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>
#include <cuda_runtime.h>
#include <cuda.h>


#define UCT_CUDA_IPC_TL_NAME    "cuda_ipc"
#define UCT_CUDA_IPC_DEV_NAME   "cudaipc0"

#define CUDA_FUNC(func)  ({                             \
ucs_status_t _status = UCS_OK;                          \
do {                                                    \
    CUresult _result = (func);                          \
    if (CUDA_SUCCESS != _result) {                      \
        ucs_error("[%s:%d] cuda failed with %d \n",     \
         __FILE__, __LINE__,_result);                   \
        _status = UCS_ERR_IO_ERROR;                     \
    }                                                   \
} while (0);                                            \
_status;                                                \
})

typedef struct uct_cuda_ipc_iface {
    uct_base_iface_t        super;
    ucs_mpool_t             cuda_event_desc;
    ucs_queue_head_t        pending_event_q;
    cudaStream_t            stream_d2h;
    cudaStream_t            stream_h2d;
    cudaStream_t            stream_d2d;
    ucs_callback_t          progress;
    int                     cuda_own_dev_num; /*
                                               * Assuming that just
                                               * one device will be
                                               * used per interface
                                               */
} uct_cuda_ipc_iface_t;

typedef struct uct_cuda_ipc_iface_config {
    uct_iface_config_t      super;
} uct_cuda_ipc_iface_config_t;

typedef struct uct_cuda_ipc_event_desc {
    cudaEvent_t event;
    uct_completion_t *comp;
    ucs_queue_elem_t  queue;
} uct_cuda_ipc_event_desc_t;

#endif
