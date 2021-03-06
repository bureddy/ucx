/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/arch/bitops.h>
#include "ib_mlx5.h"
#include "ib_mlx5_log.h"
#include "ib_mlx5_ifc.h"

#if HAVE_DEVX
typedef struct uct_ib_mlx5_mem {
    uct_ib_mem_t             super;
    struct mlx5dv_devx_obj   *atomic_dvmr;
} uct_ib_mlx5_mem_t;
#else
typedef struct uct_ib_mem uct_ib_mlx5_mem_t;
#endif

typedef struct uct_ib_mlx5_dbrec_page {
    struct mlx5dv_devx_umem *mem;
} uct_ib_mlx5_dbrec_page_t;


#if HAVE_DECL_MLX5DV_INIT_OBJ
ucs_status_t uct_ib_mlx5dv_init_obj(uct_ib_mlx5dv_t *obj, uint64_t type)
{
    int ret;

    ret = mlx5dv_init_obj(&obj->dv, type);
#if HAVE_IBV_EXP_DM
    if (!ret && (type & MLX5DV_OBJ_DM)) {
        ret = uct_ib_mlx5_get_dm_info(obj->dv_dm.in, obj->dv_dm.out);
    }
#endif
    if (ret != 0) {
        ucs_error("DV failed to get mlx5 information. Type %lx.", type);
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}
#endif


static ucs_status_t uct_ib_mlx5dv_memh_reg(uct_ib_md_t *ibmd,
                                           uct_ib_mem_t *ib_memh,
                                           off_t offset)
{
#if HAVE_DEVX
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);
    uint32_t out[UCT_IB_MLX5DV_ST_SZ_DW(create_mkey_out)] = {};
    struct ibv_mr *mr = memh->super.mr;
    ucs_status_t status = UCS_OK;
    struct mlx5dv_pd dvpd = {};
    struct mlx5dv_obj dv = {};
    size_t reg_length, length, inlen;
    int list_size, i;
    void *mkc, *klm;
    uint32_t *in;
    intptr_t addr;

    if (!(md->flags & UCT_IB_MLX5_MD_FLAG_KSM)) {
        return UCS_ERR_UNSUPPORTED;
    }

    reg_length = UCT_IB_MD_MAX_MR_SIZE;
    addr       = (intptr_t)mr->addr & ~(reg_length - 1);
    length     = mr->length + (intptr_t)mr->addr - addr;
    list_size  = ucs_div_round_up(length, reg_length);
    inlen      = UCT_IB_MLX5DV_ST_SZ_BYTES(create_mkey_in) +
                 UCT_IB_MLX5DV_ST_SZ_BYTES(klm) * list_size;

    in         = ucs_calloc(1, inlen, "mkey mailbox");
    if (in == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    dv.pd.in   = md->super.pd;
    dv.pd.out  = &dvpd;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_PD);

    UCT_IB_MLX5DV_SET(create_mkey_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_MKEY);
    mkc = UCT_IB_MLX5DV_ADDR_OF(create_mkey_in, in, memory_key_mkey_entry);
    UCT_IB_MLX5DV_SET(mkc, mkc, access_mode_1_0, UCT_IB_MLX5_MKC_ACCESS_MODE_KSM);
    UCT_IB_MLX5DV_SET(mkc, mkc, a, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, rw, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, rr, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, lw, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, lr, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, pd, dvpd.pdn);
    UCT_IB_MLX5DV_SET(mkc, mkc, translations_octword_size, list_size);
    UCT_IB_MLX5DV_SET(mkc, mkc, log_entity_size, ucs_ilog2(reg_length));
    UCT_IB_MLX5DV_SET(mkc, mkc, qpn, 0xffffff);
    UCT_IB_MLX5DV_SET(mkc, mkc, mkey_7_0, offset & 0xff);
    UCT_IB_MLX5DV_SET64(mkc, mkc, start_addr, addr + offset);
    UCT_IB_MLX5DV_SET64(mkc, mkc, len, length);
    UCT_IB_MLX5DV_SET(create_mkey_in, in, translations_octword_actual_size, list_size);

    klm = UCT_IB_MLX5DV_ADDR_OF(create_mkey_in, in, klm_pas_mtt);
    for (i = 0; i < list_size; i++) {
        if (i == list_size - 1) {
            UCT_IB_MLX5DV_SET(klm, klm, byte_count, length % reg_length);
        } else {
            UCT_IB_MLX5DV_SET(klm, klm, byte_count, reg_length);
        }
        UCT_IB_MLX5DV_SET(klm, klm, mkey, mr->lkey);
        UCT_IB_MLX5DV_SET64(klm, klm, address, addr + (i * reg_length));
        klm += UCT_IB_MLX5DV_ST_SZ_BYTES(klm);
    }

    memh->atomic_dvmr = mlx5dv_devx_obj_create(md->super.dev.ibv_context, in, inlen,
                                               out, sizeof(out));
    if (memh->atomic_dvmr == NULL) {
        ucs_debug("CREATE_MKEY KSM failed: %m");
        status = UCS_ERR_UNSUPPORTED;
        md->flags &= ~UCT_IB_MLX5_MD_FLAG_KSM;
        goto out;
    }

    memh->super.atomic_rkey =
        (UCT_IB_MLX5DV_GET(create_mkey_out, out, mkey_index) << 8) |
        (offset & 0xff);

    ucs_debug("KSM registered memory %p..%p offset 0x%lx on %s rkey 0x%x",
              mr->addr, mr->addr + mr->length, offset, uct_ib_device_name(&md->super.dev),
              memh->super.atomic_rkey);
out:
    ucs_free(in);
    return status;
#else
    return uct_ib_verbs_reg_atomic_key(ibmd, ib_memh, offset);
#endif
}

static ucs_status_t uct_ib_mlx5dv_memh_dereg(uct_ib_md_t *ibmd,
                                             uct_ib_mem_t *ib_memh)
{
#if HAVE_DEVX
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    int ret;

    ret = mlx5dv_devx_obj_destroy(memh->atomic_dvmr);
    if (ret != 0) {
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
#else
    return uct_ib_verbs_dereg_atomic_key(ibmd, ib_memh);
#endif
}

#if HAVE_DEVX

static ucs_status_t uct_ib_mlx5_add_page(ucs_mpool_t *mp, size_t *size_p, void **page_p)
{
    uct_ib_mlx5_md_t *md = ucs_container_of(mp, uct_ib_mlx5_md_t, dbrec_pool);
    uintptr_t ps = ucs_get_page_size();
    uct_ib_mlx5_dbrec_page_t *page;
    size_t size = ucs_align_up(*size_p + sizeof(*page), ps);

    page = ucs_memalign(ps, size, "devx dbrec page");
    if (page == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    page->mem = mlx5dv_devx_umem_reg(md->super.dev.ibv_context, page, size,
                                     IBV_ACCESS_LOCAL_WRITE);
    if (page->mem == NULL) {
        goto err;
    }

    *size_p = size;
    *page_p = page + 1;
    return UCS_OK;

err:
    ucs_free(page);
    return UCS_ERR_IO_ERROR;
}

static void uct_ib_mlx5_init_dbrec(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_ib_mlx5_dbrec_page_t *page = chunk - sizeof(*page);
    uct_ib_mlx5_dbrec_t *dbrec     = obj;

    dbrec->mem_id = page->mem->umem_id;
    dbrec->offset = obj - chunk + sizeof(*page);
}

static void uct_ib_mlx5_free_page(ucs_mpool_t *mp, void *chunk)
{
    uct_ib_mlx5_dbrec_page_t *page = chunk - sizeof(*page);
    mlx5dv_devx_umem_dereg(page->mem);
    ucs_free(page);
}

static ucs_mpool_ops_t uct_ib_mlx5_dbrec_ops = {
    .chunk_alloc   = uct_ib_mlx5_add_page,
    .chunk_release = uct_ib_mlx5_free_page,
    .obj_init      = uct_ib_mlx5_init_dbrec,
    .obj_cleanup   = NULL
};

static ucs_status_t uct_ib_mlx5dv_md_open(struct ibv_device *ibv_device,
                                          uct_ib_md_t **p_md)
{
    uint32_t out[UCT_IB_MLX5DV_ST_SZ_DW(query_hca_cap_out)] = {};
    uint32_t in[UCT_IB_MLX5DV_ST_SZ_DW(query_hca_cap_in)] = {};
    struct mlx5dv_context_attr dv_attr = {};
    ucs_status_t status = UCS_OK;
    int atomic = 0;
    struct ibv_context *ctx;
    uct_ib_device_t *dev;
    uct_ib_mlx5_md_t *md;
    void *cap;
    int ret;

#if HAVE_DECL_MLX5DV_IS_SUPPORTED
    if (!mlx5dv_is_supported(ibv_device)) {
        return UCS_ERR_UNSUPPORTED;
    }
#endif

    dv_attr.flags |= MLX5DV_CONTEXT_FLAGS_DEVX;
    ctx = mlx5dv_open_device(ibv_device, &dv_attr);
    if (ctx == NULL) {
        ucs_debug("mlx5dv_open_device(%s) failed: %m", ibv_get_device_name(ibv_device));
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    md = ucs_calloc(1, sizeof(*md), "ib_mlx5_md");
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context;
    }

    dev              = &md->super.dev;
    dev->ibv_context = ctx;

    IBV_EXP_DEVICE_ATTR_SET_COMP_MASK(&dev->dev_attr);
    ret = ibv_query_device_ex(dev->ibv_context, NULL, &dev->dev_attr);
    if (ret != 0) {
        ucs_error("ibv_query_device() returned %d: %m", ret);
        status = UCS_ERR_IO_ERROR;
        goto err_free;
    }

    cap = UCT_IB_MLX5DV_ADDR_OF(query_hca_cap_out, out, capability);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, opcode, UCT_IB_MLX5_CMD_OP_QUERY_HCA_CAP);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod, UCT_IB_MLX5_HCA_CAP_OPMOD_GET_MAX |
                                                   (UCT_IB_MLX5_CAP_GENERAL << 1));
    ret = mlx5dv_devx_general_cmd(ctx, in, sizeof(in), out, sizeof(out));
    if (ret == 0) {
        if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, dct)) {
            dev->flags |= UCT_IB_DEVICE_FLAG_DC;
        }
        if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, compact_address_vector)) {
            dev->flags |= UCT_IB_DEVICE_FLAG_AV;
        }
        if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, fixed_buffer_size)) {
            md->flags |= UCT_IB_MLX5_MD_FLAG_KSM;
        }
        if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, atomic)) {
            atomic = 1;
        }
    } else if ((errno != EPERM) &&
               (errno != EPROTONOSUPPORT) &&
               (errno != EOPNOTSUPP)) {
        ucs_error("MLX5_CMD_OP_QUERY_HCA_CAP failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free;
    } else {
        status = UCS_ERR_UNSUPPORTED;
        goto err_free;
    }

    status = ucs_mpool_init(&md->dbrec_pool, 0,
                            sizeof(uct_ib_mlx5_dbrec_t), 0,
                            UCS_SYS_CACHE_LINE_SIZE,
                            ucs_get_page_size() / UCS_SYS_CACHE_LINE_SIZE - 1,
                            UINT_MAX, &uct_ib_mlx5_dbrec_ops, "devx dbrec");
    if (status != UCS_OK) {
        goto err_free;
    }

    if (atomic) {
        int ops = UCT_IB_MLX5_ATOMIC_OPS_CMP_SWAP |
                  UCT_IB_MLX5_ATOMIC_OPS_FETCH_ADD;
        uint8_t arg_size;
        int cap_ops, mode8b;

        UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod, UCT_IB_MLX5_HCA_CAP_OPMOD_GET_MAX |
                                                       (UCT_IB_MLX5_CAP_ATOMIC << 1));
        ret = mlx5dv_devx_general_cmd(ctx, in, sizeof(in), out, sizeof(out));
        if (ret != 0) {
            ucs_error("MLX5_CMD_OP_QUERY_HCA_CAP failed: %m");
            return UCS_ERR_IO_ERROR;
        }

        arg_size = UCT_IB_MLX5DV_GET(atomic_caps, cap, atomic_size_qp);
        cap_ops  = UCT_IB_MLX5DV_GET(atomic_caps, cap, atomic_operations);
        mode8b   = UCT_IB_MLX5DV_GET(atomic_caps, cap, atomic_req_8B_endianness_mode);

        if ((cap_ops & ops) == ops) {
            dev->atomic_arg_sizes = sizeof(uint64_t);
            if (!mode8b) {
                dev->atomic_arg_sizes_be = sizeof(uint64_t);
            }
        }

        ops |= UCT_IB_MLX5_ATOMIC_OPS_MASKED_CMP_SWAP |
               UCT_IB_MLX5_ATOMIC_OPS_MASKED_FETCH_ADD;

        arg_size &= UCT_IB_MLX5DV_GET(query_hca_cap_out, out,
                                      capability.atomic_caps.atomic_size_dc);

        if ((cap_ops & ops) == ops) {
            dev->ext_atomic_arg_sizes = arg_size;
            if (mode8b) {
                arg_size &= ~(sizeof(uint64_t));
            }
            dev->ext_atomic_arg_sizes_be = arg_size;
        }

        dev->pci_fadd_arg_sizes  = UCT_IB_MLX5DV_GET(atomic_caps, cap, fetch_add_pci_atomic) << 2;
        dev->pci_cswap_arg_sizes = UCT_IB_MLX5DV_GET(atomic_caps, cap, compare_swap_pci_atomic) << 2;
    }

    dev->flags |= UCT_IB_DEVICE_FLAG_MLX5_PRM;
    md->flags |= UCT_IB_MLX5_MD_FLAG_DEVX;
    *p_md = &md->super;
    return status;

err_free:
    ucs_free(md);
err_free_context:
    ibv_close_device(ctx);
err:
    return status;
}

void uct_ib_mlx5dv_md_cleanup(uct_ib_md_t *ibmd)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);

    ucs_mpool_cleanup(&md->dbrec_pool, 1);
}

ucs_status_t uct_ib_mlx5_get_compact_av(uct_ib_iface_t *iface, int *compact_av)
{
    *compact_av = !!(uct_ib_iface_device(iface)->flags & UCT_IB_DEVICE_FLAG_AV);
    return UCS_OK;
}

#else

static ucs_status_t uct_ib_mlx5_check_dc(uct_ib_device_t *dev)
{
    ucs_status_t status = UCS_OK;
#if HAVE_DC_DV
    struct ibv_srq_init_attr srq_attr = {};
    struct ibv_context *ctx = dev->ibv_context;
    struct ibv_qp_init_attr_ex qp_attr = {};
    struct mlx5dv_qp_init_attr dv_attr = {};
    struct ibv_qp_attr attr = {};
    struct ibv_srq *srq;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    int ret;

    pd = ibv_alloc_pd(ctx);
    if (pd == NULL) {
        ucs_error("ibv_alloc_pd() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    cq = ibv_create_cq(ctx, 1, NULL, NULL, 0);
    if (cq == NULL) {
        ucs_error("ibv_create_cq() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_cq;
    }

    srq_attr.attr.max_sge   = 1;
    srq_attr.attr.max_wr    = 1;
    srq = ibv_create_srq(pd, &srq_attr);
    if (srq == NULL) {
        ucs_error("ibv_create_srq() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_srq;
    }

    qp_attr.send_cq              = cq;
    qp_attr.recv_cq              = cq;
    qp_attr.qp_type              = IBV_QPT_DRIVER;
    qp_attr.comp_mask            = IBV_QP_INIT_ATTR_PD;
    qp_attr.pd                   = pd;
    qp_attr.srq                  = srq;

    dv_attr.comp_mask            = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
    dv_attr.dc_init_attr.dct_access_key = UCT_IB_KEY;

    /* create DCT qp successful means DC is supported */
    qp = mlx5dv_create_qp(ctx, &qp_attr, &dv_attr);
    if (qp == NULL) {
        goto err_qp;
    }

    attr.qp_state        = IBV_QPS_INIT;
    attr.port_num        = 1;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ  |
                           IBV_ACCESS_REMOTE_ATOMIC;
    ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE |
                                   IBV_QP_PKEY_INDEX |
                                   IBV_QP_PORT |
                                   IBV_QP_ACCESS_FLAGS);
    if (ret != 0) {
        goto err;
    }

    attr.qp_state                  = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_256;
    attr.ah_attr.port_num = 1;

    ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE |
                                   IBV_QP_MIN_RNR_TIMER |
                                   IBV_QP_AV |
                                   IBV_QP_PATH_MTU);

    if (ret == 0) {
        dev->flags |= UCT_IB_DEVICE_FLAG_DC;
    }

err:
    ibv_destroy_qp(qp);
err_qp:
    ibv_destroy_srq(srq);
err_srq:
    ibv_destroy_cq(cq);
err_cq:
    ibv_dealloc_pd(pd);
#elif HAVE_DECL_IBV_EXP_DEVICE_DC_TRANSPORT && HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_EXP_DEVICE_CAP_FLAGS
    if (dev->dev_attr.exp_device_cap_flags & IBV_EXP_DEVICE_DC_TRANSPORT) {
        dev->flags |= UCT_IB_DEVICE_FLAG_DC;
    }
#endif
    return status;
}

static ucs_status_t uct_ib_mlx5dv_md_open(struct ibv_device *ibv_device,
                                          uct_ib_md_t **p_md)
{
    ucs_status_t status = UCS_OK;
    struct ibv_context *ctx;
    uct_ib_device_t *dev;
    uct_ib_mlx5_md_t *md;
    int ret;

    ctx = ibv_open_device(ibv_device);
    if (ctx == NULL) {
        ucs_debug("ibv_open_device(%s) failed: %m", ibv_get_device_name(ibv_device));
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    md = ucs_calloc(1, sizeof(*md), "ib_mlx5_md");
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context;
    }

    dev              = &md->super.dev;
    dev->ibv_context = ctx;

    IBV_EXP_DEVICE_ATTR_SET_COMP_MASK(&dev->dev_attr);
#if HAVE_DECL_IBV_EXP_QUERY_DEVICE
    ret = ibv_exp_query_device(dev->ibv_context, &dev->dev_attr);
#elif HAVE_DECL_IBV_QUERY_DEVICE_EX
    ret = ibv_query_device_ex(dev->ibv_context, NULL, &dev->dev_attr);
#else
#  error accel TL missconfigured
#endif
    if (ret != 0) {
        ucs_error("ibv_query_device() returned %d: %m", ret);
        status = UCS_ERR_IO_ERROR;
        goto err_free;
    }

    status = uct_ib_mlx5_check_dc(dev);
    if (status != UCS_OK) {
        goto err_free;
    }

    dev->flags |= UCT_IB_DEVICE_FLAG_MLX5_PRM;
    *p_md = &md->super;
    return status;

err_free:
    ucs_free(md);
err_free_context:
    ibv_close_device(ctx);
err:
    return status;
}

void uct_ib_mlx5dv_md_cleanup(uct_ib_md_t *ibmd) { }

#endif

static uct_ib_md_ops_t uct_ib_mlx5dv_md_ops = {
    .open             = uct_ib_mlx5dv_md_open,
    .cleanup          = uct_ib_mlx5dv_md_cleanup,
    .memh_struct_size = sizeof(uct_ib_mlx5_mem_t),
    .reg_atomic_key   = uct_ib_mlx5dv_memh_reg,
    .dereg_atomic_key = uct_ib_mlx5dv_memh_dereg,
};

UCT_IB_MD_OPS(uct_ib_mlx5dv_md_ops, 1);

int uct_ib_mlx5dv_arm_cq(uct_ib_mlx5_cq_t *cq, int solicited)
{
    uint64_t doorbell, sn_ci_cmd;
    uint32_t sn, ci, cmd;

    sn  = cq->cq_sn & 3;
    ci  = cq->cq_ci & 0xffffff;
    cmd = solicited ? MLX5_CQ_DB_REQ_NOT_SOL : MLX5_CQ_DB_REQ_NOT;
    sn_ci_cmd = (sn << 28) | cmd | ci;

    cq->dbrec[UCT_IB_MLX5_CQ_ARM_DB] = htobe32(sn_ci_cmd);

    ucs_memory_cpu_fence();

    doorbell = (sn_ci_cmd << 32) | cq->cq_num;

    *(uint64_t *)((uint8_t *)cq->uar + MLX5_CQ_DOORBELL) = htobe64(doorbell);

    ucs_memory_bus_store_fence();

    return 0;
}

#if HAVE_DECL_MLX5DV_OBJ_AH
void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av)
{
    struct mlx5dv_obj  dv;
    struct mlx5dv_ah   dah;

    dv.ah.in = ah;
    dv.ah.out = &dah;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_AH);

    *av = *(dah.av);
    av->dqp_dct |= UCT_IB_MLX5_EXTENDED_UD_AV;
}
#elif !HAVE_INFINIBAND_MLX5_HW_H
void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av)
{
    ucs_bug("MLX5DV_OBJ_AH not supported");
}
#endif

