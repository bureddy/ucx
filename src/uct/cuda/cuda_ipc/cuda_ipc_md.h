/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 *
 * Copyright (c) 2016-2017, NVIDIA Corporation.  All rights reserved.
 * See COPYRIGHT for license information
 */

#ifndef UCT_CUDA_IPC_H
#define UCT_CUDA_IPC_H

#include <uct/base/uct_md.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define UCT_CUDA_IPC_MD_NAME           "cuda_ipc"

extern uct_md_component_t uct_cuda_ipc_md_component;

/**
 * @brief CUDA_IPC MD descriptor
 */
typedef struct uct_cuda_ipc_md {
    struct uct_md super;   /**< Domain info */
} uct_cuda_ipc_md_t;

/**
 * @brief CUDA_IPC domain configuration.
 */
typedef struct uct_cuda_ipc_md_config {
    uct_md_config_t super;

} uct_cuda_ipc_md_config_t;

/**
 * @brief CUDA_IPC packed and remote key
 */
typedef struct uct_cuda_ipc_key {
    cudaIpcMemHandle_t ipc_mem_handle;
    uintptr_t own_addr;
    uintptr_t remote_addr;
} uct_cuda_ipc_key_t;

#endif
