/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cuda_ipc_cache.h"
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/sys.h>

static ucs_pgt_dir_t *uct_cuda_ipc_cache_pgt_dir_alloc(const ucs_pgtable_t *pgtable)
{
    return ucs_memalign(UCS_PGT_ENTRY_MIN_ALIGN, sizeof(ucs_pgt_dir_t),
                        "cuda_ipc_cache_pgdir");
}

static void uct_cuda_ipc_cache_pgt_dir_release(const ucs_pgtable_t *pgtable,
                                               ucs_pgt_dir_t *dir)
{
    ucs_free(dir);
}

ucs_status_t uct_cuda_ipc_create_cache(uct_cuda_ipc_cache_t **cache)
{
    ucs_status_t status;
    uct_cuda_ipc_cache_t *cache_desc;
    int ret;

    cache_desc = ucs_malloc(sizeof(uct_cuda_ipc_cache_t), "uct_cuda_ipc_cache_t");
    if (cache_desc == NULL) {
        ucs_error("failed to allocate memory for cuda_ipc cache");
        return UCS_ERR_NO_MEMORY;
    }

    ret = pthread_rwlock_init(&cache_desc->lock, NULL);
    if (ret) {
        ucs_error("pthread_rwlock_init() failed: %m");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    status = ucs_pgtable_init(&cache_desc->pgtable,
                              uct_cuda_ipc_cache_pgt_dir_alloc,
                              uct_cuda_ipc_cache_pgt_dir_release);
    if (status != UCS_OK) {
        goto err_destroy_rwlock;
    }

    *cache = cache_desc;
    return UCS_OK;

err_destroy_rwlock:
    pthread_rwlock_destroy(&cache_desc->lock);
err:
    free(cache_desc);
    return status;
}

static void
uct_cuda_ipc_cache_region_collect_callback(const ucs_pgtable_t *pgtable,
                                           ucs_pgt_region_t *pgt_region,
                                           void *arg)
{
    ucs_list_link_t *list = arg;
    uct_cuda_ipc_cache_region_t *region;

    region = ucs_derived_of(pgt_region, uct_cuda_ipc_cache_region_t);
    ucs_list_add_tail(list, &region->list);
}

static void uct_cuda_ipc_cache_purge(uct_cuda_ipc_cache_t *cache)
{
    uct_cuda_ipc_cache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_list_head_init(&region_list);
    ucs_pgtable_purge(&cache->pgtable, uct_cuda_ipc_cache_region_collect_callback,
                      &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        UCT_CUDADRV_FUNC(cuIpcCloseMemHandle((CUdeviceptr)region->mapped_addr));
        ucs_free(region);
    }
    ucs_trace("cuda ipc cache purged");
}

void uct_cuda_ipc_destroy_cache(uct_cuda_ipc_cache_t *cache)
{
    uct_cuda_ipc_cache_purge(cache);
    ucs_pgtable_cleanup(&cache->pgtable);
    pthread_rwlock_destroy(&cache->lock);
    ucs_free(cache);
}

static ucs_status_t uct_cuda_ipc_openmemhandle(CUipcMemHandle memh,
                                               CUdeviceptr *mapped_addr)
{
    CUresult cuerr;

    cuerr = cuIpcOpenMemHandle(mapped_addr, memh,
                               CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
    if (cuerr != CUDA_SUCCESS) {
        if (cuerr == CUDA_ERROR_ALREADY_MAPPED) {
            return UCS_ERR_ALREADY_EXISTS;
        }
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static void uct_cuda_ipc_cache_invalidate_regions(uct_cuda_ipc_cache_t *cache,
                                            void *from, void *to)
{
    ucs_list_link_t region_list;
    ucs_status_t status;
    uct_cuda_ipc_cache_region_t *region, *tmp;

    ucs_list_head_init(&region_list);
    ucs_pgtable_search_range(&cache->pgtable, (ucs_pgt_addr_t)from,
                             (ucs_pgt_addr_t)to,
                             uct_cuda_ipc_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        status = ucs_pgtable_remove(&cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_error("failed to remove address:%p from cache",
                      (void *)region->key.d_bptr);
        }
        UCT_CUDADRV_FUNC(cuIpcCloseMemHandle((CUdeviceptr)region->mapped_addr));
        ucs_free(region);
    }
    ucs_trace("closed memhandles in the range [%p..%p]", from, to);
}

ucs_status_t uct_cuda_ipc_cache_map_memhandle(void *arg, uct_cuda_ipc_key_t *key,
                                              void **mapped_addr)
{
    uct_cuda_ipc_cache_t *cache = (uct_cuda_ipc_cache_t *)arg;
    ucs_status_t status;
    ucs_pgt_region_t *pgt_region;
    uct_cuda_ipc_cache_region_t *region;

    pthread_rwlock_rdlock(&cache->lock);
    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup,
                                  &cache->pgtable, key->d_bptr);
    if (ucs_likely(pgt_region != NULL)) {
        region = ucs_derived_of(pgt_region, uct_cuda_ipc_cache_region_t);
        if (memcmp((const void *)&key->ph, (const void *)&region->key.ph,
                   sizeof(key->ph)) == 0) {
            /*cache hit */
            ucs_trace("cuda_ipc cache hit addr:%p size:%lu",
                      (void *)key->d_bptr, key->b_len);

            *mapped_addr = region->mapped_addr;
            pthread_rwlock_unlock(&cache->lock);
            return UCS_OK;
        } else {
            ucs_trace("cuda_ipc cache found stale handle. "
                      " old_addr:%p old_size:%lu new_addr:%p new_size:%lu",
                      (void *)region->key.d_bptr, region->key.b_len,
                      (void *)key->d_bptr, key->b_len);

            status = ucs_pgtable_remove(&cache->pgtable, &region->super);
            if (status != UCS_OK) {
                ucs_error("failed to remove address:%p from cache",
                          (void *)key->d_bptr);
                goto err;
            }

            /* close memhandle */
            UCT_CUDADRV_FUNC(cuIpcCloseMemHandle((CUdeviceptr)
                                                 region->mapped_addr));
            ucs_free(region);
        }
    }

    status = uct_cuda_ipc_openmemhandle(key->ph, (CUdeviceptr *)mapped_addr);
    if (ucs_unlikely(status != UCS_OK)) {
        if (ucs_likely(status == UCS_ERR_ALREADY_EXISTS)) {
            /* unmap all overlapping regions and retry*/
            uct_cuda_ipc_cache_invalidate_regions(cache, (void *)key->d_bptr,
                                                  (void *)key->d_bptr + key->b_len);
            status = uct_cuda_ipc_openmemhandle(key->ph, (CUdeviceptr *)mapped_addr);
            if (ucs_unlikely(status != UCS_OK)) {
                if (ucs_likely(status == UCS_ERR_ALREADY_EXISTS)) {
                    /* unmap all cache entries and retry */
                    uct_cuda_ipc_cache_purge(cache);
                    status = uct_cuda_ipc_openmemhandle(key->ph, (CUdeviceptr *)mapped_addr);
                    if (status != UCS_OK) {
                        ucs_fatal("failed to open ipc mem handle. addr:%p len:%lu",
                                  (void *)key->d_bptr, key->b_len);
                    }
                } else {
                    ucs_fatal("failed to open ipc mem handle. addr:%p len:%lu",
                              (void *)key->d_bptr, key->b_len);
                }
            }
        } else {
            ucs_fatal("failed to open ipc mem handle. addr:%p len:%lu",
                      (void *)key->d_bptr, key->b_len);
        }
    }

    /*create new cache entry */
    region = ucs_memalign(UCS_PGT_ENTRY_MIN_ALIGN,
                          sizeof(uct_cuda_ipc_cache_region_t),
                          "uct_cuda_ipc_cache_region");
    if (region == NULL) {
        ucs_warn("failed to allocate uct_cuda_ipc_cache region");
        goto err;
    }

    region->super.start = ucs_align_down_pow2((uintptr_t)key->d_bptr,
                                               UCS_PGT_ADDR_ALIGN);
    region->super.end   = ucs_align_up_pow2  ((uintptr_t)key->d_bptr + key->b_len,
                                               UCS_PGT_ADDR_ALIGN);
    region->key         = *key;
    region->mapped_addr = *mapped_addr;

    status = UCS_PROFILE_CALL(ucs_pgtable_insert,
                              &cache->pgtable, &region->super);
    if (status != UCS_OK) {
        ucs_error("failed to insert region " UCS_PGT_REGION_FMT ": %s",
                  UCS_PGT_REGION_ARG(&region->super), ucs_status_string(status));
        ucs_free(region);
        goto err;
    }

    ucs_trace("cuda_ipc cache new enrtry. addr:%p size:%lu",
              (void *)key->d_bptr, key->b_len);

    pthread_rwlock_unlock(&cache->lock);
    return UCS_OK;
err:
    pthread_rwlock_unlock(&cache->lock);
    return status;
}
