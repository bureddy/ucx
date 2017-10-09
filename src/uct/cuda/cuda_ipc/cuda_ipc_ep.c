/**
 * Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 *
 * Copyright (c) 2016-2017, NVIDIA Corporation.  All rights reserved.
 * See COPYRIGHT for license information
 */

#include "cuda_ipc_ep.h"
#include "cuda_ipc_iface.h"
#include "cuda_ipc_md.h"

#include <uct/base/uct_log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>


static UCS_CLASS_INIT_FUNC(uct_cuda_ipc_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    uct_worker_progress_add_safe(iface->super.worker,
                                 iface->progress, iface,
                                 &iface->super.prog);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_ipc_ep_t)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_cuda_ipc_iface_t);

    uct_worker_progress_remove(iface->super.worker, &iface->super.prog);
}

UCS_CLASS_DEFINE(uct_cuda_ipc_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_ipc_ep_t, uct_ep_t, uct_iface_t*,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_ipc_ep_t, uct_ep_t);

ucs_status_t uct_cuda_ipc_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                       size_t iovcnt, uint64_t remote_addr,
                                       uct_rkey_t rkey, uct_completion_t *comp)
{
    ucs_status_t status;
    uct_cuda_ipc_iface_t *iface           = NULL;
    uct_cuda_ipc_event_desc_t *cuda_event = NULL;
    uct_cuda_ipc_key_t *ipc_key           = (uct_cuda_ipc_key_t *) rkey;
    int offset                            = -1;
    void *src                             = NULL;

    iface = ucs_derived_of(tl_ep->iface, uct_cuda_ipc_iface_t);
    cuda_event = ucs_mpool_get(&iface->cuda_event_desc);
    offset = (uintptr_t) remote_addr - ipc_key->remote_addr;
    src = (char *) ipc_key->own_addr + offset;

    status = CUDA_FUNC(cudaMemcpyAsync(iov[0].buffer, (void *)src, iov[0].length,
                                       cudaMemcpyDeviceToDevice,
                                       iface->stream_d2d));
    if (UCS_OK != status) {
        ucs_error("cudaMemcpyAsync Failed ");
        return UCS_ERR_IO_ERROR;
    }

    status = CUDA_FUNC(cudaEventRecord(cuda_event->event, iface->stream_d2d));
    if (UCS_OK != status) {
        ucs_error("cudaEventRecord Failed ");
        return UCS_ERR_IO_ERROR;
    }
    cuda_event->comp = comp;

    ucs_queue_push(&iface->pending_event_q, &cuda_event->queue);

    ucs_info("cuda async issued :%p buffer:%p  len:%ld", cuda_event,
             iov[0].buffer, iov[0].length);

    return UCS_INPROGRESS;

}

ucs_status_t uct_cuda_ipc_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                       size_t iovcnt, uint64_t remote_addr,
                                       uct_rkey_t rkey, uct_completion_t *comp)
{
    ucs_status_t status;
    uct_cuda_ipc_iface_t *iface           = NULL;
    uct_cuda_ipc_event_desc_t *cuda_event = NULL;
    uct_cuda_ipc_key_t *ipc_key           = (uct_cuda_ipc_key_t *) rkey;
    int offset                            = -1;
    void *dst                             = NULL;

    iface = ucs_derived_of(tl_ep->iface, uct_cuda_ipc_iface_t);
    cuda_event = ucs_mpool_get(&iface->cuda_event_desc);
    offset = (uintptr_t) remote_addr - ipc_key->remote_addr;
    dst = (char *) ipc_key->own_addr + offset;

    status = CUDA_FUNC(cudaMemcpyAsync(dst, iov[0].buffer, iov[0].length,
                                       cudaMemcpyDeviceToDevice,
                                       iface->stream_d2d));
    if (UCS_OK != status) {
        ucs_error("cudaMemcpyAsync Failed ");
        return UCS_ERR_IO_ERROR;
    }

    status = CUDA_FUNC(cudaEventRecord(cuda_event->event, iface->stream_d2d));
    if (UCS_OK != status) {
        ucs_error("cudaEventRecord Failed ");
        return UCS_ERR_IO_ERROR;
    }
    cuda_event->comp = comp;

    ucs_queue_push(&iface->pending_event_q, &cuda_event->queue);

    ucs_info("cuda async issued :%p buffer:%p  len:%ld", cuda_event,
             iov[0].buffer, iov[0].length);

    return UCS_INPROGRESS;

}

